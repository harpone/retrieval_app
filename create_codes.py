import numpy as np
import pandas as pd
import time
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch.multiprocessing as mp
import torch
from os.path import join
from torch.utils.data import DataLoader
import uuid
from termcolor import colored

from core.augs import load_augs, mask_to_fixed_shape
from core.models import SuperModel
from core.dataio import Database, image_from_url, upload_to_gcs
from core.datasets import URLDataset
from core.utils import visualize_segmentations, compute_visual_center
from core.config import RESIZE_TO

"""

- about 1.1 s if max shorter = 256 which is OK

"""

# TODO: maybe also line simplification to constant number of points: https://github.com/Permafacture/Py-Visvalingam-Whyatt

# for sanity checks:
catalog = MetadataCatalog.get('coco_2017_train_panoptic_separated')
thing_classes = catalog.thing_classes
stuff_classes = catalog.stuff_classes


def create_codes(gpu,
                 image_urls,
                 db_out_folder,
                 db_out_basename,
                 num_workers=1,
                 upload_to_storage=False,
                 add_random_hash=False):
    """Generates a retrieval code and coarse grained segmentation for each image in `image_paths` and inserts these
    into an hdf5 store together with the image filename. Store will be saved to `out_path`

    :param image_urls: list of strings
    :param db_out_folder: base path where hdf5 store will be saved
    :return:
    """
    # profiler = Profiler()
    # profiler.start()

    torch.cuda.set_device(gpu)
    num_gpus = torch.cuda.device_count()

    # Define storage:
    flush_every = 1000
    # h5file = tb.open_file(out_path, mode="w", title='OpenImages database, COCO labels for seg, ImageNet for image')
    url_max_len = image_urls.dtype.itemsize
    # TODO: getting url_max_len from image_urls is fine when square byte array, but in a streaming scenario probably
    # need to set a sufficiently high number, e.g. 128
    expectedrows = len(image_urls) // num_gpus

    # TODO: replace str(gpu) with random string
    if add_random_hash:
        db_out_name = db_out_basename + '_' + uuid.uuid1().hex[:16] + '.h5'
    else:
        db_out_name = db_out_basename + '.h5'
    database = Database(join(db_out_folder, db_out_name), url_max_len=url_max_len, mode='w', title=None, expected_rows=expectedrows)

    # Load models:
    # print('Loading models.')
    # models = load_models()  # about 13s
    # segnet = models['segnet']
    # segnet.model.cuda(gpu)
    # repnet = models['repnet'].cuda(gpu)
    # pca = models['pca']
    supermodel = SuperModel()

    # Def augs:
    augs = load_augs(resize_to=RESIZE_TO)

    # Def dataloader:
    image_urls_this = image_urls[gpu::num_gpus]  # split evenly for all devices
    dataset = URLDataset(url_list=image_urls_this, transform=None)  # TODO: now None!!

    def drop_batch_dim(x_):
        return x_[0]
    dataloader = DataLoader(
                dataset,
                batch_size=1,  # because batching done in dataset
                num_workers=num_workers,
                collate_fn=drop_batch_dim,
                pin_memory=True
            )

    counter_images = 0
    counter_codes = 0
    start_time = time.time()
    print('Begin generating codes.')
    with torch.no_grad():  # TODO: need this?
        for img, image_url, shape_orig in dataloader:
            counter_images += 1
            if img is None or img.mode != 'RGB':  # still PIL but transformed
                continue
            # shape_current = np.array(list(img.size))  # w, h
            # counter_images += 1
            #
            # # Compute rescaling factor from original and current image sizes:
            # aspect_orig = shape_orig[0] / shape_orig[1]
            # aspect_current = shape_current[0] / shape_current[1]
            # rescaler = aspect_orig / aspect_current

            # # Segmentation:
            # outputs_seg = segnet(augs['augs_seg'](img))
            # seg, segments_info = outputs_seg["panoptic_seg"]
            # # visualize_segmentations(augs['augs_seg'](img), seg, segments_info)  # TODO: comment when done debugging
            # seg_masks = np.eye(seg.max() + 1)[seg.cpu().numpy()].transpose(2, 0, 1).astype(
            #     bool)  # [num_segs, H_floor32, W_floor32]
            #
            # # Representation:
            # logits, codes = repnet(augs['augs_rep'](img)[None].cuda())  # e.g. inp: [256, 416] out: [8192, 8, 13]
            # pred_img = logits[0].argmax().cpu()  # TODO: sanity check several of these
            # codes = codes[0].detach().cpu().numpy()  # e.g. shape [8192, 8, 13]
            #
            # #### Get global code and meta:
            # code = codes.mean(-1).mean(-1)  # [8192, ]
            # code = pca.transform(code[None])  # [1, 128]

            results = supermodel(img)

            for id_, result_this in results.items():
                counter_codes += 1
                del result_this['seg_mask']  # will not be stored for now
                database.append_to_store(url=image_url, **result_this)

                if counter_codes % flush_every == 0:
                    database.flush()

            # # Append: (visual center zero by def, no item pred)
            # database.append_to_store(image_url, code, pred_img)
            #
            # # Resize segmentation to repnet shape:
            # seg_masks_small = zoom(seg_masks, [1, 1 / 32, 1 / 32], order=0)  # e.g. shape [num_segs, 8, 13]
            # for seg_mask_small, seg_mask, seg_info in zip(seg_masks_small[1:], seg_masks[1:], segments_info):
            #     # TODO: now including also non-things, but maybe filter or post process later
            #     counter_codes += 1
            #
            #     #### Get local code and meta:
            #     seg_mask_area = seg_mask_small.sum()
            #     code = (seg_mask_small[None] * codes).sum(-1).sum(-1) / (seg_mask_area + 1e-8)  # [8192, ]
            #     code = models['pca'].transform(code[None])  # [1, 128]
            #     pred_item = seg_info['category_id']  # corresponds to `catalog` categories (thing or stuff)
            #
            #     # Visual center from large seg_mask:
            #     h_center, w_center = compute_visual_center(seg_mask)
            #
            #     # Adjust visual center to original image coords:
            #     # Note that now shorter side is always size RESIZE_TO and therefore not cropped
            #     if shape_orig[0] / shape_orig[1]:  # landscape: cropped in w
            #         w_center /= rescaler
            #     else:  # portrait: cropped in h
            #         h_center *= rescaler
            #
            #     # Append:
            #     database.append_to_store(image_url,
            #                              code,
            #                              pred_img,
            #                              h_center=h_center,
            #                              w_center=w_center,
            #                              pred_item=pred_item,
            #                              is_thing=seg_info['isthing'],
            #                              global_code=False)
            #
            #     if counter_codes % flush_every == 0:
            #         database.flush()

            print(f'\r{db_out_folder.split("/")[-1]}: images={counter_images}, codes={counter_codes} '
                  f':: {round(time.time() - start_time)} 'f'seconds', end='')

            # profiling:
            # if counter_images == 100:
            #    break

    database.close()

    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))

    if upload_to_storage:
        remote_path = join('databases', db_out_name)
        upload_to_gcs('neodatasets', db_out_folder, remote_path)
        print(f'Results uploaded to {remote_path}')
    return True


if __name__ == '__main__':

    limit_to = 10000
    num_gpus = 1
    num_workers = 4
    upload_to_storage = False

    #urls_path = 'https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train0.tsv'
    urls_path = 'https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-validation.tsv'
    db_out_folder = f'/home/heka/database/'

    db_out_basename = urls_path.split('/')[-1].split('.')[0]

    print(colored('Downloading urls from online...', 'yellow'))
    df = pd.read_csv(urls_path, sep='\t', index_col=False, usecols=['TsvHttpData-1.0'])
    image_urls_o = df['TsvHttpData-1.0'].values  # original size image urls

    # "thumbnails" instead of original:
    image_urls_z = [url.replace('_o.jpg', '_z.jpg') for url in image_urls_o]
    image_urls_z = np.array(image_urls_z).astype(np.string_)[:limit_to]

    os.makedirs(db_out_folder, exist_ok=True)

    if num_gpus > 1:
        add_random_hash = True
        mp.spawn(create_codes,
                 args=(image_urls_z, db_out_folder, db_out_basename, num_workers, upload_to_storage, add_random_hash))

    else:
        add_random_hash = False
        create_codes(0, image_urls_z, db_out_folder, db_out_basename, num_workers, upload_to_storage, add_random_hash)
