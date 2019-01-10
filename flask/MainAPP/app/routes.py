from flask import render_template
from app import app

@app.route('/')
@app.route('/index')
def index():
    user = {'username':'User'}
    posts = [
        {
            'author':{'username':'John'},
            'body':'Beatiful day in Granada'
        },
        {
            'author':{'username':'Susan'},
            'body':'The Avengers movie was so cool'
        }
    ]
    return render_template('index.html', title='Home', user=user, posts=posts)

@app.route('/upload')
def upload():
    user = {'username':'User'}
    return render_template('upload.html', title='Upload', user=user)
