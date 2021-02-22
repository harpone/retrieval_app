import numpy as np
from flask import Flask, render_template, request, url_for, redirect, Response, flash
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from werkzeug.exceptions import abort
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import os
import cv2
from termcolor import colored
from PIL import Image
from waitress import serve
import ngtpy
from flask_bootstrap import Bootstrap
from flask_dropzone import Dropzone


from core.dataio import Database
from core.models import SuperModel
from core.augs import load_augs
from core.config import RESIZE_TO, N_RETRIEVED_RESULTS
from core.utils import get_query_plot, get_retrieval_plot, delete_plot_cache

DEBUGGING_WITHOUT_MODEL = False
DEBUG_WITH_PREDS = False  # will show image, item preds in plots

# TODO: could be a bad idea using locals in the first place... multiprocessing could get these confused?
RESULTS = None
query_img_path = None
retrieval_img_path = None  # not yet retrieved
ids = None
#uploaded_filename = None
uploaded_image = None


"""
TODO:
- now debugging locally with old database!!!! Rebuild index - takes lots or RAM locally
- build small dev db/index
"""

app = Flask(__name__)
#app.config['SECRET_KEY'] = 'asdfhbas7f3f3qoah'
app.config.from_pyfile('configs/appconfig.py')
app.config.update(
    UPLOADED_PHOTOS_DEST='./static/cache',
    UPLOADED_PATH='./static/cache',
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=2,
    DROPZONE_MAX_FILES=1,
    DROPZONE_REDIRECT_VIEW='query_image'  # set redirect view
)
bootstrap = Bootstrap(app)
dropzone = Dropzone(app)


photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, u'Image only!'), FileRequired(u'File was empty!')])
    submit = SubmitField(u'Upload')


# Delete cache folder:
delete_plot_cache()

# Set up video capture:
videocap = cv2.VideoCapture(-1)
print(colored('Video capture device initialized', 'green'))

# Set up database:  # TODO: protect codes and index! Needs refactoring!! Actually maybe
#database_name = 'open-images-dataset-train0_0_475000.h5'  # TODO: as arg maybe
#database_name = 'db_jan_2021b.h5'  # newest
database_name = 'dev_db.h5'  # for local dev & debugging
database_root = '/home/heka/model_data'
database = Database(database_name, data_root=database_root, mode='r')
codes = database.codes
entities = database.table  # use .table for retrieval, table.row for insertion

# Build index if one doesn't exist:
index_name = database_name.split('.')[0] + '_index'
index_path = os.path.join('/home/heka/model_data', index_name)
if not os.path.exists(index_path):
    print(colored('Creating NGTPY index for the first time. '
                  'This can take a while (around 1s per 10k objects)...', 'green'))
    if not os.path.exists('/home/heka/model_data/'):
        os.makedirs('/home/heka/model_data/', exist_ok=True)
    ngtpy.create(path=index_path, dimension=128, object_type='Float')
    ngtpy_index = ngtpy.Index(index_path)
    codes = np.array_split(codes, 10)  # TODO: smarter split
    for n, code in enumerate(codes):
        print(f'\r  batch {n}', end='')
        ngtpy_index.batch_insert(np.array(code, dtype=np.float64), num_threads=8)  # 11s for 100k objects;
    ngtpy_index.save()
else:
    print(colored('Loading an existing NGTPY index...', 'green'))
    ngtpy_index = ngtpy.Index(index_path)

# Load Supermodel:
if DEBUGGING_WITHOUT_MODEL:
    supermodel = None
else:
    print(colored('Loading model...', 'green'))
    supermodel = SuperModel()

# Def augs:
augs = load_augs(resize_to=RESIZE_TO)

print('READY TO FLY!')


def get_numpy_frame():
    _, frame = videocap.read()  # frame [480, 640, 3] by default  # TODO: sometimes returns frame = None!
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def get_jpeg_frame():
    _, frame = videocap.read()  # frame [480, 640, 3] by default
    _, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()


def get_entity(idx):
    try:
        _, entity = database[idx]
    except IndexError:
        abort(404)
    return entity


def generate_feed():
    while True:
        jpeg = get_jpeg_frame()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n'


@app.route('/', methods=['GET', 'POST'])
def index():
    global uploaded_image
    if request.method == 'POST':
        f = request.files.get('file')
        uploaded_image = Image.open(f).convert('RGB')
        #return redirect(url_for('query_image'))  # don't redirect - flask-dropzone does that for some reason
    return render_template('index.html')


@app.route('/show_feed')
def show_feed():
    if videocap.isOpened():
        return render_template('show_feed.html')
    else:
        print('TRYING TO USE WEBCAM')
        flash('Not implemented yet...')
        return redirect(url_for('index'))


@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/query_image', methods=['GET', 'POST'])
def query_image():

    global RESULTS
    global query_img_path
    global retrieval_img_path
    global ids
    global uploaded_image

    if 'back' in request.form:
        delete_plot_cache()
        RESULTS = None
        query_img_path = None
        retrieval_img_path = None
        ids = None
        return redirect(url_for('index'))
    elif any(['entity' in key for key in request.form.keys()]):  # query entities or entire image
        item_id = eval(list(request.form.keys())[0].split('_')[-1])  # TODO: check that getting correct value!
        #code, h_center, w_center, pred, isthing, seg_mask = RESULTS[item_id]
        img_meta = RESULTS[item_id]

        # Search for nns:
        query_results = ngtpy_index.search(img_meta['code'], N_RETRIEVED_RESULTS)
        indices, _ = list(zip(*query_results))

        retrieval_img_path = get_retrieval_plot(indices, entities, debug_mode=DEBUG_WITH_PREDS)
    elif uploaded_image is not None:  # uploaded photo
        #img = Image.open(os.path.join('./static/cache', uploaded_filename)).convert('RGB')
        query_img_base64, ids = process_image(uploaded_image)
        uploaded_image = None
    return render_template('query_image.html',
                           query_img=query_img_base64,
                           ids=ids,
                           retrieval_img_path=retrieval_img_path)


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


def process_image(img_):
    """

    :param img_: PIL Image
    :return:
    """
    global RESULTS

    # Resize if image is huge:
    size_overflow = np.max(img_.size) / 1024
    if size_overflow > 1:
        w, h = img_.size
        w = int(w / size_overflow)
        h = int(h / size_overflow)
        img_ = img_.resize((w, h), resample=2)

    img_aug = augs['augs_base'](img_)  # [256, .., 3] or [.., 256, 3]; stil PIL

    # supermodel out:
    if DEBUGGING_WITHOUT_MODEL:  # debugging
        results_load = np.load('supermodel_out.npz', allow_pickle=True)
        RESULTS = {int(key): results_load[key] for key in results_load.files}
    else:
        RESULTS = supermodel(
            img_)  # dict with items [code, h_center, w_center, pred, isthing, seg_mask]; 0 is global

    # bake in the segmentations to the PIL image:
    query_img_base64 = get_query_plot(img_, img_aug, RESULTS, debug_mode=DEBUG_WITH_PREDS)

    # entity ids for HTML:
    labels = ['Image']
    labels += [str(i + 1) for i in range(len(RESULTS.keys()) - 1)]
    ids = {label: 'entity_' + str(num) for label, num in zip(labels, np.arange(len(labels)))}

    return query_img_base64, ids


@app.route('/<int:idx>')
def post(idx):
    entity = get_entity(idx)
    return render_template('entity.html', entity=entity)


if __name__ == '__main__':
    # TODO: should I be using argparse? Maybe not...
    # TODO: arg for database to be used
    # TODO: switch for reset index
    # TODO: def stuff like /home/heka/model_data/ in configs

    # parser = ArgumentParser(add_help=False)
    # parser.add_argument(
    #     "--database-path",
    #     default='/home/heka/database/test_50k',
    #     type=str,
    # )
    # args = parser.parse_args()

    #app.run(debug=False)
    this_files_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(this_files_dir)
    serve(app, host='0.0.0.0', port=8001, threads=1)  # TODO: check threads OK!
