import numpy as np
import pandas as pd
import time
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch.multiprocessing as mp
import torch
from itertools import islice
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
        #for img, image_url, shape_orig in dataloader:
        dataloader_iterator = islice(iter(dataloader), 3)
        while True:
            try:
                img, image_url, shape_orig = next(dataloader_iterator)
            except StopIteration:
                break
            except Exception as e:  # need to skip if very rare error (pytorch urllib3.exceptions.ProtocolError)
                print(e)
                continue

            counter_images += 1
            if img is None or img.mode != 'RGB':  # still PIL but transformed
                continue

            results = supermodel(img)

            for id_, result_this in results.items():
                counter_codes += 1
                del result_this['seg_mask']  # will not be stored for now
                database.append_to_store(url=image_url, **result_this)

                if counter_codes % flush_every == 0:
                    database.flush()

            print(f'\r{db_out_folder.split("/")[-1]}: images={counter_images}, codes={counter_codes} '
                  f':: {round(time.time() - start_time)} 'f'seconds', end='')

    database.close()

    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))

    if upload_to_storage:
        remote_path = join('databases', db_out_name)
        upload_to_gcs('mldata-westeu', join(db_out_folder, db_out_name), remote_path)
        print(f'\nResults uploaded to {remote_path}')
    return True


if __name__ == '__main__':

    start_from = 100000
    end_at = 225000
    num_gpus = 1
    num_workers = 4
    upload_to_storage = True

    print('**************************')
    print(f'start_from={start_from}')
    print(f'end_at={end_at}')
    print(f'num_workers={num_workers}')
    print('**************************')

    urls_path = 'https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train0.tsv'  # about 1 min
    #urls_path = 'https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-validation.tsv'
    db_out_folder = f'./database/'

    db_out_basename = urls_path.split('/')[-1].split('.')[0] + '_' + str(start_from) + '_' + str(end_at)

    print(colored('Downloading urls from online...', 'yellow'), end='')
    df = pd.read_csv(urls_path, sep='\t', index_col=False, usecols=['TsvHttpData-1.0'])
    print(colored('... done!', 'yellow'))
    image_urls_o = df['TsvHttpData-1.0'].values  # original size image urls

    # "thumbnails" instead of original:
    image_urls_z = [url.replace('_o.jpg', '_z.jpg') for url in image_urls_o]
    image_urls_z = np.array(image_urls_z).astype(np.string_)[start_from:end_at]

    os.makedirs(db_out_folder, exist_ok=True)

    if num_gpus > 1:
        add_random_hash = True
        mp.spawn(create_codes,
                 args=(image_urls_z, db_out_folder, db_out_basename, num_workers, upload_to_storage, add_random_hash))

    else:
        add_random_hash = False
        create_codes(0, image_urls_z, db_out_folder, db_out_basename, num_workers, upload_to_storage, add_random_hash)
