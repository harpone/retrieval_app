import os
import matplotlib.pyplot as plt
from os.path import join
import webdataset as wds
import json
from PIL import Image
import pandas as pd
import numpy as np
from multiprocessing import Process

import sys

sys.path.append(os.getcwd())
import core.utils as utils

"""
- about 10 examples/ s for single process => 1000 ex/s for 100 processes => about 30 min total
- about 33 MB for 100 examples => about 561 GB total (if original size)
- assumes masks are in a specific local folder - all other data (incl. images) pulled from online
"""


def write_tar(pid,
              image_meta_pid,
              df_masks,
              df_labels,
              df_bboxes,
              df_relations):
    # TODO: not memory efficient to load same data in each process... maybe use shm if memory problems

    seg_idx2cls = pd.read_csv(SEG_CLASSES_URL, header=None).to_dict()[0]
    seg_cls2idx = {val: key for key, val in seg_idx2cls.items()}

    # Def output tar and TarWriter:
    tar_name = TAR_BASENAME + f'-{pid}.tar'
    tar_path = join(PATH_ROOT, tar_name)
    sink = wds.TarWriter(tar_path, encoder=False)  # note: encoder=False now
    counter = 0
    counter_missing = 0
    for _, (img_id, url) in image_meta_pid.iterrows():
        if IMAGE_SIZE != 'o':  # instead use lower res image
            url = url.replace('_o.jpg', f'_{IMAGE_SIZE}.jpg')

        img_bytes = utils.image_bytes_from_url(url)  # note: bytes, so use turbodecoder or pildecoder
        if img_bytes is None:
            counter_missing += 1
            continue
        #img.load()  # load here because lazy op
        counter += 1
        print(f'\r{counter}', end='')

        # Load masks and process:
        df_masks_this = df_masks[df_masks.ImageID == img_id]  # TODO: slow for big dataframes
        mask = combine_masks(df_masks_this, seg_cls2idx, img_bytes)  # uint16 array; returns empty mask if no segs

        # Compress to bytes with blosc:
        mask_bytes = utils.compress_to_bytes(mask)

        # Load labels:
        labels_df_this = df_labels[df_labels.ImageID == img_id]
        targets = labels_df_this[labels_df_this.columns[2:]].to_dict('list')
        targets['LabelPresence'] = targets.pop('Confidence')
        targets['LabelNameImage'] = targets.pop('LabelName')

        # Load bboxes:
        bboxes_df_this = df_bboxes[df_bboxes.ImageID == img_id]
        bboxes = bboxes_df_this[[bboxes_df_this.columns[2]] + list(bboxes_df_this.columns[4:])].to_dict('list')
        bboxes['LabelNameBB'] = bboxes.pop('LabelName')

        targets.update(bboxes)

        # Load relations:
        relations_df_this = df_relations[df_relations.ImageID == img_id]
        relations = relations_df_this[relations_df_this.columns[1:]].to_dict('list')

        targets.update(relations)

        # To bytes:
        targets = bytes(json.dumps(targets), encoding='utf-8')

        # Pack to sample:
        sample = {
            '__key__': img_id,
            'image.jpg': img_bytes,
            'mask.png': mask_bytes,
            'targets.json': targets,
        }
        sink.write(sample)

        # if counter % 100 == 0:
        #     print(f'\r{counter}', end='')
        #     break  # testing

    sink.close()
    print()
    print(f'pid={pid}: not found={counter_missing}')


def combine_masks(df_masks_, cls2idx, img_bytes):
    try:
        img = utils.turbodecoder(img_bytes)
    except OSError:  # "Unsupported color conversion request"?
        img = utils.pildecoder(img_bytes)
    mask_canvas = np.zeros(img.shape[:2], dtype=np.uint16)
    for _, row in df_masks_.iterrows():
        mask_fname = row.MaskPath
        mask = Image.open(join(MASKS_PATH, mask_fname))  # PNG
        label_name = row.LabelName
        label_int = cls2idx[label_name]
        mask = mask.resize(mask_canvas.shape[::-1])

        mask_canvas[np.array(mask)] = label_int

    return mask_canvas


if __name__ == '__main__':

    """
    NOTE: flickr image sizes (longer side)
    o: original
    z: 640
    c: 800  # AOK
    l: 1024  # not working!
    """

    #PATH_ROOT = '/mnt/disks/datasets/openimages'
    PATH_ROOT = '/media/heka/TERA/Data/openimages'

    num_samples_per_tarfile = 3000
    SEG_CLASSES_URL = 'https://storage.googleapis.com/openimages/v5/classes-segmentation.txt'
    IMAGE_SIZE = 'z'

    if 1:  # openimages validation set
        # Note: getting only 24730 masks. but that's correct
        print('Creating validation set tars.')
        MASKS_PATH = join(PATH_ROOT, 'val/masks')
        IMGS_URL = 'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv'
        MASKS_URL = 'https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv'
        LABELS_URL = 'https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv'
        BBOXES_URL = 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv'
        RELATIONS_URL = 'https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-vrd.csv'
        TAR_BASENAME = f'val/openimages-{IMAGE_SIZE}-val'
        num_tar_files = 1

    elif 0:  # training set
        print('Creating training set tars.')
        MASKS_PATH = join(PATH_ROOT, 'train/masks')
        IMGS_URL = 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
        MASKS_URL = 'https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv'
        LABELS_URL = 'https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv'
        BBOXES_URL = 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv'
        RELATIONS_URL = 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-vrd.csv'
        TAR_BASENAME = f'train/openimages-{IMAGE_SIZE}-train'
        num_tar_files = 64

    else:  # test set
        print('Creating test set tars.')
        MASKS_PATH = join(PATH_ROOT, 'test/masks')
        IMGS_URL = 'https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv'
        MASKS_URL = 'https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv'
        LABELS_URL = 'https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels-boxable.csv'
        BBOXES_URL = 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv'
        RELATIONS_URL = 'https://storage.googleapis.com/openimages/v6/oidv6-test-annotations-vrd.csv'
        TAR_BASENAME = f'test/openimages-{IMAGE_SIZE}-test'
        num_tar_files = 32

    print('Loading metadata from URL:')
    df_masks = pd.read_csv(MASKS_URL)
    df_labels = pd.read_csv(LABELS_URL)
    df_bboxes = pd.read_csv(BBOXES_URL)
    df_relations = pd.read_csv(RELATIONS_URL)
    image_meta = pd.read_csv(IMGS_URL)
    image_meta = image_meta[['ImageID', 'OriginalURL']]
    image_meta = image_meta.sample(frac=1).reset_index(drop=True)  # shuffle now so no need to shuffle later

    print(f'Splitting data into {num_tar_files} tar archives:')
    image_meta_split = np.array_split(image_meta, num_tar_files)

    # Split all dataframes according to image id split:
    if 1:  # multiprocess
        processes = []
        for pid, image_meta_pid in enumerate(image_meta_split):
            print(f'\rInitializing process {pid}', end='')
            df_masks_pid = df_masks[df_masks.ImageID.isin(image_meta_pid.ImageID)]
            df_labels_pid = df_labels[df_labels.ImageID.isin(image_meta_pid.ImageID)]
            df_bboxes_pid = df_bboxes[df_bboxes.ImageID.isin(image_meta_pid.ImageID)]
            df_relations_pid = df_relations[df_relations.ImageID.isin(image_meta_pid.ImageID)]
            processes.append(Process(target=write_tar, args=(pid,
                                                             image_meta_pid,
                                                             df_masks_pid,
                                                             df_labels_pid,
                                                             df_bboxes_pid,
                                                             df_relations_pid)))
        print()
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        print('DONE!')
    else:
        write_tar(0, image_meta_split[0])
