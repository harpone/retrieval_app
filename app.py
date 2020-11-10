from argparse import ArgumentParser
import numpy as np
import ngtpy
from flask import Flask, render_template, request, url_for, flash, redirect, Response
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from werkzeug.exceptions import abort
import torch
import base64
import io
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
from termcolor import colored
from PIL import Image

from core.dataio import Database
from core.models import SuperModel
from core.augs import load_augs
from core.config import RESIZE_TO
from core.utils import fuse_results

RESULTS = None
DEBUGGING_WITHOUT_MODEL = True
N_RETRIEVED_RESULTS = 5

"""

"""
# TODO: maybe all globals in uppercase?
app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdfhbas7f3f3qoah'


# Set up video capture:
videocap = cv2.VideoCapture(0)
print(colored('Video capture device initialized', 'green'))

# Set up database:  # TODO: protect codes and index! Needs refactoring!! Actually maybe
database = Database('/home/heka/database/test_50k.h5', mode='r')
codes = database.codes

# Build index if one doesn't exist:
index_path = './model_data/ngtpy_index'
if not os.path.exists(index_path):
    print(colored('Creating NGTPY index for the first time. '
                  'This can take a while (around 1s per 10k objects)...', 'green'))
    ngtpy.create(path=index_path, dimension=128, object_type='Float')
    ngtpy_index = ngtpy.Index(index_path)
    ngtpy_index.batch_insert(np.array(codes))  # TODO: limits of batch_insert?  11s for 100k objects @home
    ngtpy_index.save()
else:
    print(colored('Loading an existing NGTPY index...', 'green'))
    ngtpy_index = ngtpy.Index(index_path)

# for sanity checks:
catalog = MetadataCatalog.get('coco_2017_train_panoptic_separated')
thing_classes = catalog.thing_classes
stuff_classes = catalog.stuff_classes

# Load Supermodel:
if DEBUGGING_WITHOUT_MODEL:
    supermodel = None
else:
    supermodel = SuperModel()

# Def augs:
augs = load_augs(resize_to=RESIZE_TO)


def get_numpy_frame():
    ret, frame = videocap.read()  # frame [480, 640, 3] by default  # TODO: sometimes returns frame = None!
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def get_jpeg_frame():
    ret, frame = videocap.read()  # frame [480, 640, 3] by default
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()


def get_entity(idx):
    try:
        code, entity = database[idx]
    except IndexError:
        abort(404)
        entity = None
    return entity


def generate_feed():
    while True:
        jpeg = get_jpeg_frame()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n'


@app.route('/', methods=['GET', 'POST'])
def index():
    # codes = list()
    # entities = list()
    # for _ in range(3):
    #     idx = np.random.randint(100000)
    #     code, entity = database[idx]
    #     codes.append(code)
    #     entity['idx'] = idx
    #     entities.append(entity)
    # return render_template('index.html', entities=entities)
    if 'stop' in request.form:
        print('Taking picture...')
        return redirect(url_for('query_image'))
    elif 'start' in request.form:
        print('Starting video feed...')
        return redirect(url_for('show_feed'))
    else:
        return render_template('index.html')


@app.route('/show_feed')
def show_feed():
    return render_template('show_feed.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/query_image', methods=['GET', 'POST'])
def query_image():
    global RESULTS
    if 'restart' in request.form:
        print('Starting video feed...')
        return redirect(url_for('show_feed'))
    elif any(['entity' in key for key in request.form.keys()]):  # query entities or entire image
        item_id = eval(list(request.form.keys())[0].split('_')[-1])
        code, h_center, w_center, pred, isthing, seg_mask = RESULTS[item_id]

        # Search for nns:
        query_results = ngtpy_index.search(code, N_RETRIEVED_RESULTS)
        indices, dists = list(zip(*query_results))

        # Get corresponding entities from database:





        print(item_id)
        return redirect(url_for('query_image'))
    else:  # will just redisplay original snapshot
        img = get_numpy_frame()  # [480, 640, 3] uint8 by default
        img_orig = Image.fromarray(img)
        img_aug = augs['augs_base'](img_orig)  # [256, .., 3] or [.., 256, 3]; stil PIL

        # supermodel out:
        if DEBUGGING_WITHOUT_MODEL:  # debugging
            results_load = np.load('supermodel_out.npz', allow_pickle=True)
            RESULTS = {int(key): results_load[key] for key in results_load.files}
        else:
            RESULTS = supermodel(img_aug)  # dict with items [code, h_center, w_center, pred, isthing, seg_mask]; 0 is global

        # bake in the segmentations to the PIL image:
        query_img_path = fuse_results(img_orig, img_aug, RESULTS)
        #query_img_path = os.path.abspath(query_img_path)

        # entity ids for HTML:
        labels = ['Image']
        labels += [str(i + 1) for i in range(len(RESULTS.keys()) - 1)]
        ids = {label: 'entity_' + str(num) for label, num in zip(labels, np.arange(len(labels)))}

        return render_template('query_image.html', query_img_path=query_img_path, ids=ids)


@app.route('/<int:idx>')
def post(idx):
    entity = get_entity(idx)
    return render_template('entity.html', entity=entity)


if __name__ == '__main__':
    # TODO: should I be using argparse? Maybe not...
    # parser = ArgumentParser(add_help=False)
    # parser.add_argument(
    #     "--database-path",
    #     default='/home/heka/database/test_50k',
    #     type=str,
    # )
    # args = parser.parse_args()

    app.run(debug=False)
