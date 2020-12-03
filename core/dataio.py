import requests
from PIL import Image
import numpy as np
import tables as tb
import os
import cv2
from os.path import join
from google.cloud import storage
from multiprocessing import Pool
from termcolor import colored
import time

from core.config import CODE_LENGTH


def get_dataloader(args, phase='train'):
    # TODO
    return


def blob_to_path(bucketname, blob_path=None, local_path=None):

    store = storage.Client()
    bucket = store.bucket(bucketname)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)


def capture_webcam():
    """Takes one photo with webcam.

    :return:
    """
    video_capture = cv2.VideoCapture(0)
    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")

    _, frame = video_capture.read()  # Read picture. ret === True on success

    # Close device
    video_capture.release()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


def upload_to_gcs(bucketname, blob_path=None, local_path=None):
    """Upload a file to GCS.

    """
    store = storage.Client()
    bucket = store.bucket(bucketname)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

    return


class Database:

    def __init__(self, database_name, url_max_len=128, mode='w', title=None, expected_rows=1000000):

        self.data_root = '/home/heka/model_data/'
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root, exist_ok=True)

        # if 'gs:' in database_name:  # get from cloud storage  # TODO: shit, refactor or delete
        #     store = storage.Client()
        #     bucket = store.bucket('mldata-westeu')
        #     database_name = database_name.split('mldata-westeu')[-1]  # TODO fuck this is ugly
        #     blob = bucket.blob(database_name[1:])
        #     blob.download_to_filename(database_name)

        class Entity(tb.IsDescription):
            """Metadata for a given item. Aligned with `code_arr` and `segmask_arr`.

            url: image url
            h_center: visual center height in relative coords (in [0, 1]) of item segmentation
            w_center:
            global_code: bool; True if image level global code
            """
            url = tb.StringCol(url_max_len)
            h = tb.Float16Col()
            w = tb.Float16Col()
            pred = tb.StringCol(32)
            is_thing = tb.BoolCol()

        if mode == 'w':
            self.h5file = tb.open_file(join(self.data_root, database_name), mode=mode, title=title)

            # Schema:
            self.table = self.h5file.create_table(self.h5file.root,
                                                  'entities',
                                                  Entity,
                                                  "Entity metadata",
                                                  expectedrows=expected_rows)
            self.entities = self.table.row
            self.codes = self.h5file.create_earray(self.h5file.root,
                                                   'codes',
                                                   atom=tb.Float16Atom(),
                                                   shape=(0, CODE_LENGTH),
                                                   expectedrows=expected_rows)

        else:
            try:
                print(colored(f'Trying to load database from {join(self.data_root, database_name)}'))
                self.h5file = tb.open_file(join(self.data_root, database_name), mode=mode)
            except:  # TODO: catch? OSError at least maybe also read error
                print(colored('Local database not found... downloading from GCS.', 'red'))
                blob_to_path('mldata-westeu',
                             blob_path=join('databases', database_name),
                             local_path=join(self.data_root, database_name))
                time.sleep(1)
                self.h5file = tb.open_file(join(self.data_root, database_name), mode=mode)

            self.table = self.h5file.root.entities
            self.entities = self.table.row
            self.codes = self.h5file.root.codes

        self.table_keys = list(self.table.coldescrs.keys())

    def append_to_store(self,
                        url=None,
                        code=None,
                        h=0,
                        w=0,
                        pred=None,
                        is_thing=False):
        self.codes.append(code.astype(np.float16))
        self.entities['url'] = url
        self.entities['h'] = float(h)
        self.entities['w'] = float(w)
        self.entities['pred'] = pred
        self.entities['is_thing'] = is_thing
        self.entities.append()

    def cat(self, other):
        # TODO: about 1 min for 600k codes... faster way?
        # TODO: how does perf suffer because expectedrows?
        i = 0
        while True:
            try:
                code, entity = other[i]
                self.append_to_store(str(entity['url'], encoding='utf-8'),
                                     code[None],
                                     h=entity['h'],
                                     w=entity['w'],
                                     pred=entity['pred'],
                                     is_thing=entity['is_thing'])
                i += 1
            except TypeError:  # TODO: catch
                raise
            except Exception as e:
                if 'out of range' in e.args[0]:  # TODO: ugly
                    print('Done.')
                    break
                else:
                    raise e

    def flush(self):
        self.h5file.flush()

    def close(self):
        self.h5file.close()

    def __getitem__(self, i):

        code = self.codes[i]
        entity_list = list(self.table[i])
        entity = {key: val for key, val in zip(self.table_keys, entity_list)}

        return code, entity


def image_from_url(url):
    """Load image from `url`

    :param url: str
    :return:
    """
    r = requests.get(url, stream=True)
    if r.status_code == 200:  # AOK
        r.raw.decode_content = True
        img = Image.open(r.raw)
    else:
        img = None

    return img


def images_from_urls(urls, num_processes=None):
    """Load multiple images from a list of urls in parallel.

    :param urls: list of strings
    :param num_processes: 1 or None; will use all available processes if None
    :return:
    """

    if num_processes == 1:
        images = [image_from_url(url) for url in urls]
    elif num_processes is None:
        with Pool() as pool:
            images = pool.map(image_from_url, urls)
    else:
        raise NotImplementedError

    return images
