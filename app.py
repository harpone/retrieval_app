from argparse import ArgumentParser
import numpy as np
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort

from core.dataio import Database

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdfhbas7f3f3qoah'

database = Database('/home/heka/database/test_50k.h5', mode='r')

def get_entity(idx):
    try:
        code, entity = database[idx]
    except IndexError:
        abort(404)
        entity = None
    return entity


@app.route('/')
def index():
    codes = list()
    entities = list()
    for _ in range(3):
        idx = np.random.randint(100000)
        code, entity = database[idx]
        codes.append(code)
        entity['idx'] = idx
        entities.append(entity)
    return render_template('index.html', entities=entities)


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




    app.run()
