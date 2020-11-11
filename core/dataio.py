import requests
from PIL import Image
import numpy as np
import tables as tb
import os
import cv2
from os.path import join
from google.cloud import storage
from multiprocessing import Pool, cpu_count

from core.config import CODE_LENGTH

def capture_webcam():
    """Takes one photo with webcam.

    :return:
    """
    video_capture = cv2.VideoCapture(0)
    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")

    ret, frame = video_capture.read()  # Read picture. ret === True on success

    # Close device
    video_capture.release()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


def upload_to_gcs(bucketname, local_path, remote_path):
    """Upload a file to GCS.

    """
    store = storage.Client()
    bucket = store.bucket(bucketname)
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_path)

    return


class Database:

    def __init__(self, database_path, url_max_len=128, mode='w', title=None, expected_rows=1000000):

        # try:
        #     entities = self.h5file.root.entities
        #     url_max_len = entities.coldtypes['url'].itemsize
        #     print('`url_max_len` overrided from opened database!')
        # except:  # TODO: catch
        #     entities = None
        #     pass

        class Entity(tb.IsDescription):
            """Metadata for a given item. Aligned with `code_arr` and `segmask_arr`.

            url: image url
            h_center: visual center height in relative coords (in [0, 1]) of item segmentation
            w_center:
            global_code: bool; True if image level global code
            """
            url = tb.StringCol(url_max_len)
            h_center = tb.Float16Col()
            w_center = tb.Float16Col()
            global_code = tb.BoolCol()
            prediction_image = tb.Int32Col()
            prediction_item = tb.Int32Col()
            is_thing = tb.BoolCol()

        if mode == 'w':
            self.h5file = tb.open_file(database_path, mode=mode, title=title)

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
            self.h5file = tb.open_file(database_path, mode=mode)
            self.table = self.h5file.root.entities
            self.entities = self.table.row
            self.codes = self.h5file.root.codes

        self.table_keys = list(self.table.coldescrs.keys())

    def append_to_store(self,
                        url_,
                        code_,
                        pred_img_,
                        h_center=0,
                        w_center=0,
                        pred_item=-1,
                        is_thing=False,
                        global_code=True):
        self.codes.append(code_.astype(np.float16))
        self.entities['url'] = url_
        self.entities['prediction_image'] = pred_img_
        self.entities['h_center'] = float(h_center)
        self.entities['w_center'] = float(w_center)
        self.entities['global_code'] = global_code
        self.entities['prediction_item'] = pred_item
        self.entities['is_thing'] = is_thing
        self.entities.append()

    def cat(self, other):

        i = 0
        while True:
            try:
                code, entity = other[i]
                self.append_to_store(str(entity['url'], encoding='utf-8'),
                                     code[None],
                                     entity['prediction_image'],
                                     entity['h_center'],
                                     entity['w_center'],
                                     entity['prediction_item'],
                                     entity['is_thing'],
                                     entity['global_code'])
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
