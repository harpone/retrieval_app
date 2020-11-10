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
#supermodel = SuperModel()
supermodel = lambda x: (x, x, x)  # TODO: for debugging

# Def augs:
augs = load_augs(resize_to=RESIZE_TO)


def get_numpy_frame():
    ret, frame = videocap.read()  # frame [480, 640, 3] by default
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


@app.route('/query_image')
def query_image():
    img = get_numpy_frame()  # [480, 640, 3] uint8 by default
    img = Image.fromarray(img)
    img = augs['augs_base'](img)  # [256, .., 3] or [.., 256, 3]; stil PIL
    videocap.release()  # TODO: or not if want to retake?

    # supermodel out:
    code_global, pred_global, local_results = supermodel(img)
    # local_results = [code_local, h_center, w_center, pred_item, seg_mask] per entity; seg_mask not stored in database



    # TODO: bake in the segmentations to the PIL image

    # img to jpeg for display:
    img_io = io.BytesIO()
    img.convert('RGB').save(img_io, 'JPEG')
    img_io.seek(0)
    img = base64.b64encode(img_io.getvalue()).decode('ascii')

    return render_template('query_image.html', img=img)


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
