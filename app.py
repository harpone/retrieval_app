from argparse import ArgumentParser
import numpy as np
from flask import Flask, render_template, request, url_for, flash, redirect, Response
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from werkzeug.exceptions import abort
import cv2
from termcolor import colored

from core.dataio import Database

"""
'I was facing this problem while running the flask server in debug mode because it called cv2.VideoCapture(0) twice.'

"""

# while(True):
#
#     ret, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(32) == 27:  # ESC will stop
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# print('DONE TEEEEEEEEEEEEEEEEEEEEEEEEEST')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdfhbas7f3f3qoah'


# Set up video capture:
videocap = cv2.VideoCapture(0)
print(colored('Video capture device initialized', 'green'))

# Set up database:
database = Database('/home/heka/database/test_50k.h5', mode='r')


def get_numpy_frame():
    ret, frame = videocap.read()  # frame [480, 640, 3] by default
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
    try:
        counter = 0
        while True:
            counter += 1
            #if counter > 100:
            #    break
            jpeg = get_jpeg_frame()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')
    except KeyboardInterrupt:  # TODO: somehow not catching these exceptions...
        print(colored('Interrupted by user.', 'yellow'))
        videocap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(colored('EXCEPTION:', 'yellow'))
        videocap.release()
        cv2.destroyAllWindows()
        raise e


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

    return render_template('index.html')


@app.route('/show_feed')
def show_feed():
    return render_template('show_feed.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/query_image')
def query_image():
    return render_template('query_image.html')


@app.route('/get_query_image')
def get_query_image():
    frame = get_numpy_frame()  # [480, 640, 3] by default

    # TODO: do the ML stuff

    # To jpeg for display:
    ret, jpeg = cv2.imencode('.jpg', frame)
    jpeg = jpeg.tobytes()
    response = (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')

    return Response(response, mimetype='multipart/x-mixed-replace; boundary=frame')


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
