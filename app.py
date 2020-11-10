from argparse import ArgumentParser
import numpy as np
from flask import Flask, render_template, request, url_for, flash, redirect, Response
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from werkzeug.exceptions import abort
import torch
import base64
import io
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
from termcolor import colored
from PIL import Image

from core.dataio import Database
from core.models import SuperModel
from core.augs import load_augs
from core.config import RESIZE_TO
from core.utils import fuse_results

DEBUGGING_WITHOUT_MODEL = True

"""

"""

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdfhbas7f3f3qoah'


# Set up video capture:
videocap = cv2.VideoCapture(0)
print(colored('Video capture device initialized', 'green'))

# Set up database:
database = Database('/home/heka/database/test_50k.h5', mode='r')

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
    if 'restart' in request.form:
        print('Starting video feed...')
        return redirect(url_for('show_feed'))
    elif 'Image' in request.form:  # query for global image

        return redirect(url_for('query_image'))
    elif len(list(request.form.keys())) == 1:  # TODO hackety hacky shit!!
        item_id = int(list(request.form.keys())[0])
        print(item_id)
        return redirect(url_for('query_image'))
    else:  # will just redisplay original snapshot
        img = get_numpy_frame()  # [480, 640, 3] uint8 by default
        img_orig = Image.fromarray(img)
        img_aug = augs['augs_base'](img_orig)  # [256, .., 3] or [.., 256, 3]; stil PIL

        # supermodel out:
        if DEBUGGING_WITHOUT_MODEL:  # debugging
            results_load = np.load('supermodel_out.npz', allow_pickle=True)
            results = {int(key): results_load[key] for key in results_load.files}
        else:
            results = supermodel(img_aug)  # dict with items [code, h_center, w_center, pred, isthing, seg_mask]; 0 is global

        # bake in the segmentations to the PIL image:
        buf = fuse_results(img_orig, img_aug, results)

        # entity ids for HTML:
        ids = ['Image']
        #ids += [str(i+1) for i in range(len(results.keys()) - 1)]
        ids += [i + 1 for i in range(len(results.keys()) - 1)]

        return render_template('query_image.html', img=buf, ids=ids)


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
