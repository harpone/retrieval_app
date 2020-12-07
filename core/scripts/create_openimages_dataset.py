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

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import core.dataio as dataio
import core.utils as utils

"""
- about 10 examples/ s for single process => 1000 ex/s for 100 processes => about 30 min total
- about 33 MB for 100 examples => about 561 GB total (if original size)
- assumes masks are in a specific local folder - all other data (incl. images) pulled from online
"""
# TODO: getting UserWarning: ReadError('unexpected end of data') and Empty File etc so double check data consistency


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
    sink = wds.TarWriter(join(PATH_ROOT, tar_name), encoder=True)
    counter = 0
    counter_notfound = 0
    for _, (img_id, url) in image_meta_pid.iterrows():
        if IMAGE_SIZE != 'o':  # instead use lower res image
            url = url.replace('_o.jpg', f'_{IMAGE_SIZE}.jpg')
        counter += 1
        print(f'\r{counter}', end='')  # TODO: can get crowded...

        img = dataio.image_from_url(url)  # TODO: retrying if timeouts etc
        if img is None:
            counter_notfound += 1
            continue
        img.load()
        # img_bytes = img.tobytes()

        # Load masks and process:
        df_masks_this = df_masks[df_masks.ImageID == img_id]  # TODO: slow for big dataframes
        if len(df_masks_this) > 0:
            mask = combine_masks(df_masks_this, seg_cls2idx, img)  # PNG PIL, int32
            # mask_bytes = mask.tobytes()
        else:
            continue  # TODO: debugging

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
            'image.jpg': img,  # TODO: need bytes or will PIL work?
            'mask.png': mask,
            'targets.json': targets,
        }
        sink.write(sample)

        # if counter % 100 == 0:
        #     print(f'\r{counter}', end='')
        #     break  # testing

    sink.close()
    print(f'pid={pid}: not found={counter_notfound}')


def combine_masks(df_masks_, cls2idx, img):
    h = img.height
    w = img.width
    mask_canvas = np.zeros([h, w], dtype=np.uint32)
    for _, row in df_masks_.iterrows():
        mask_fname = row.MaskPath
        mask = Image.open(join(MASKS_PATH, mask_fname))  # PNG
        label_name = row.LabelName
        label_int = cls2idx[label_name]
        mask = mask.resize(mask_canvas.shape[::-1])

        mask_canvas[np.array(mask)] = label_int  # TODO: this will erase anything under it

    return Image.fromarray(mask_canvas)


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
    USE_ORIGINAL_SIZE = True  # TODO: False when real thing maybe
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

    elif 0:  # training set
        print('Creating training set tars.')
        MASKS_PATH = join(PATH_ROOT, 'train/masks')
        IMGS_URL = 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
        MASKS_URL = 'https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv'
        LABELS_URL = 'https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv'
        BBOXES_URL = 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv'
        RELATIONS_URL = 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-vrd.csv'
        TAR_BASENAME = f'train/openimages-{IMAGE_SIZE}-train'

    else:  # test set
        print('Creating test set tars.')
        MASKS_PATH = join(PATH_ROOT, 'test/masks')
        IMGS_URL = 'https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv'
        MASKS_URL = 'https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv'
        LABELS_URL = 'https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels-boxable.csv'
        BBOXES_URL = 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv'
        RELATIONS_URL = 'https://storage.googleapis.com/openimages/v6/oidv6-test-annotations-vrd.csv'
        TAR_BASENAME = f'test/openimages-{IMAGE_SIZE}-test'

    print('Loading metadata from URL:')
    df_masks = pd.read_csv(MASKS_URL)
    df_labels = pd.read_csv(LABELS_URL)
    df_bboxes = pd.read_csv(BBOXES_URL)
    df_relations = pd.read_csv(RELATIONS_URL)
    image_meta = pd.read_csv(IMGS_URL)
    image_meta = image_meta[['ImageID', 'OriginalURL']]
    image_meta = image_meta.sample(frac=1).reset_index(drop=True)  # shuffle now so no need to shuffle later

    # Split paths for each process:
    num_tar_files = int(len(image_meta) / num_samples_per_tarfile)
    num_tar_files = 1  # TODO: debugging and testing!
    #num_tar_files = 2  # 580 is way too many for the training set... need to rearrange later
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

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        print('DONE!')
    else:
        write_tar(0, image_meta_split[0])
