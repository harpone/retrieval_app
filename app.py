from flask import Flask, render_template

app = Flask(__name__)

posts = [dict(title='Hello'), dict(title='world')]

@app.route('/')
def index():
    return render_template('index.html', posts=posts)


if __name__ == '__main__':
    app.run()
