import os
from flask import render_template, request, flash, redirect, url_for
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from app import app
from app import models
# from app.forms import UploadForm

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

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in app.config['ALLOWD_EXTENSIONS']

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        target = os.path.join(app.config['APP_ROOT'], 'documents')
        uploaded_files = request.files.getlist('file')
        #if 'file' not in uploaded_files:
        #if uploaded_files not in request.files:
        #    flash('No file part')
        #    return redirect(url_for('index'))
        # file = request.files['file']
        #if uploaded_files == '':
        #    return redirect(url_for('index'))
        #for file in uploaded_files:
        #    if allowed_file(file.filename)
        # for file in request.files.getlist('file'):
        #if file and allowed_file(file.filename):
            # user = {'username':'User'}
        #        filename = secure_filename(file.filename)
                #destination = os.path.join(target, app.config['APP_ROOT'], filename)
                # filename = file.filename
        #        destination = "/".join([target, filename])
        #        file.save(destination)
        #        flash('File uploaded')
        #        return redirect(url_for('upload'))
        files = request.files.to_dict();
        for file in files:
            filename = files[file].filename
            if allowed_file(filename):
                nombre = secure_filename(filename)
                fileclass = File()
                fileclass.filename = nombre
                fileclass._file.put(file, content_type=file.content_type)
                fileclass.save()
                destination = "/".join([target, nombre])
                files[file].save(destination)
        return redirect(url_for('subida', filename=filename))
    return render_template('upload.html', title='Upload')


@app.route('/subida/<path:filename>')
def subida(filename):
    return render_template('subida.html', title='Subidos')
