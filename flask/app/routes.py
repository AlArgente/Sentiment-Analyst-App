import os, sys, getopt, pprint
import csv
import json
import numpy as np
import pandas as pd
from flask import render_template, request, flash, redirect, url_for, jsonify, Response, send_file, stream_with_context
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from werkzeug.datastructures import Headers
from app import app
from app import models
import gridfs
from IPython.display import display_html
from pymongo import MongoClient
from app import readwrite as rw
from app import preprocessing as pr
from app.preprocesamiento.preprocesamiento import Preprocessing as pra
from app.algorithms.algorithms import Bayes, SVM
from app import visualization as vi
# from app.prepo import preprocessing as pr
from app import dbmanager as dbm
from bson.objectid import ObjectId
import pickle
# CrossValidation unir a los algoritmos
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
# from app.forms import UploadForm

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    models, ids = dbm.get_all_models_saved(collection='PrModels')
    return render_template('index.html', title='Home', models=list(zip(models, ids))[:5])

# Se muestran todos los modelos guardados en la BD
@app.route('/pr_modelo')
def pr_modelo():
    # Obtenemos los modelos que hay guardados
    models, ids = dbm.get_all_models_saved(collection='PrModels')
    return render_template('pr_modelo.html',title='Modelos almacenados', models=list(zip(models, ids)))

# Se ven algunos datos del modelo
@app.route('/ver_modelo/<id_modelo>', methods=['POST','GET'])
def ver_modelo(id_modelo):
    model, model_data = dbm.get_model_by_id(model_id=id_modelo, collection='PrModels')
    if isinstance(model, Bayes):
        print('Instancia de Bayes')
    else:
        print('Instancia de SVM')
        # pred = jsonify(model.predict(X_test))
        # return send_file(model, mimetype='pickle/pickle')
    if request.method == 'POST':
        print('HERE')
        uploaded_files = request.files.getlist('file')
        filename = rw.read_csvs(uploaded_files)
        return redirect(url_for('testing', id_modelo=id_modelo, filename=filename))
    return render_template('ver_modelo.html', model = model,
                            model_prepro = model_data['Prepro'],
                            model_alg = model_data['Alg'],
                            model_name = model_data['nombre'],
                            model_hora = model_data['Hora'], model_lang=model_data['lang'],
                            model_t_size=model_data['t_size'], title='Modelo')

@app.route('/download_model/<id_modelo>')
def download_model_pred(id_modelo):
    resultados = dbm.get_saved_results_model(id_modelo)
    # return send_file(model, mimetype='pickle/pickle')
    tmp_name = id_modelo + '.csv'
    tmp_path = 'tmpfiles/tmp.csv'

    # return Response(csv_file, mimetype="text/csv",
    #                headers={"Content-disposition":"attachment;filename=tmp_name"})
    """headers = Headers()
    headers.set('Content-Disposition', 'attachment', filename=tmp_path)
    return Response(
        stream_with_context(csv_file),
        mimetype='text/csv', headers=headers"""
    # results =
    return send_file(tmp_path, attachment_filename='tmp.csv', as_attachment=True)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in app.config['ALLOWD_EXTENSIONS']

@app.route('/testing/<id_modelo>/<filename>')
def testing(id_modelo, filename):
    data = dbm.mongo_to_df(filename)
    model, model_data = dbm.get_model_by_id(model_id=id_modelo, collection='PrModels')
    train, test_text = pr.split_to_train_test(data)
    resultados = False
    if 'Sentiment' in data.columns:
        data_class = train["Sentiment"]
        resultados = True
    train_text = train["SentimentText"]
    # test = train_text[:1]
    test = test_text["SentimentText"]
    """
    train, test = pr.split_to_train_test(data)
    # options = request.form.getlist('preprocessing')
    train_text = train["SentimentText"]
    train_class = train["Sentiment"]
    test_text = test["SentimentText"]
    """
    print('Type of Vectorizer: ' + str(type(model_data['Features'])))
    for o in model_data['Prepro']:
        if o == 'ToLower':
            print(o)
            train_text, test = pr.to_lower(train_text, test)
        if o == 'DeleteAccent':
            print(o)
            train_text, test= pr.delete_accent(train_text, test)
        if o == 'Tokenize':
            print(o)
            train_text, test = pr.tokenize(train_text, test)
        if o == 'DeleteStopwords':
            print(o)
            train_text, test = pr.delete_stopwords(train_text, test)
        if o == 'PosTage':
            print(o)
            print('Mostrado')
        if o =='Stemmer':
            print(o)
            train_text, test = pr.stemmer(train_text, test)
        if o == 'tfidf':
            print(o)
            # data_text = pr.tfidf_devolopment(train_text, model_data['Vectorizer'])
            train_text = pr.tfidf_devolopment(train_text, model_data['Features'])
            # train_text = data_text['train']
            # test = data_text['test']
    # datos = np.concatenate((train,test))
    print('Se va a calcular la precisión del modelo')
    print(train_text.shape)
    # print(test.shape)
    print('Entro a predict')
    pred = model.predict(train_text)
    print(type(pred))
    print('Salgo de predict')
    # print(np.array(data_class))
    # print(type(pred))
    # print(type(np.array(data_class)))
    # print('Calcular accuracy_score:')
    # accu = accuracy_score(pred, np.array(data_class))
    print('Se calculan las métricas.')
    # results = vi.prec_rec_f1(y_true=data_class, y_pred=pred, average='micro')
    if resultados:
        recall = vi.recall(data_class, pred)
        precision = vi.precision(data_class, pred)
        f1score = vi.f1_sco(data_class, pred)
        print('SE VA A GUARDAR LA IMAGEN')
        vi.recall_precision_plot(data_class, pred, id_modelo)
        print('SE HA GUARDADO LA IMAGEN')
    else:
        recall = 0.0
        precision = 0.0
        f1score = 0.0

    results = {
        'Recall':recall,
        'Precision':precision,
        'F1-Score':f1score,
        'Predict': pred
    }
    print('Se han calculado las medidas.')
    for k, v in results.items():
        print(str(k) + ' ' + str(v))
    # print('Precisión del modelo: ' + str(accu))
    # print('Precisión del modelo: ' + str(model.score(pred, np.array(data_class))))
    dbm.save_results_model(id_modelo, results)

    return redirect(url_for('resultados', title="Resultados test", id_modelo=id_modelo, filename=filename))
    # return render_template('testing.html', pred=results)
#def csv_to_json(filename):

@app.route('/testing/<id_modelo>/<filename>/resultados', methods=['GET'])
def resultados(id_modelo, filename):
    resultados = dbm.get_saved_results_model(id_modelo)
    # results = resultados['results']
    df = pd.DataFrame(resultados['Predict'])
    print('MUESTRO OS.LISTDIR')
    print(os.listdir())
    print('OS.LISTDIR MOSTRADO')
    df.to_csv('app/tmpfiles/tmp.csv')
    path ='images/' + id_modelo + '.png'
    print(path)
    del resultados['Predict']
    return render_template('testing.html', pred=resultados, url=path, id_modelo=id_modelo)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        target = os.path.join(app.config['APP_ROOT'], 'documents')
        uploaded_files = request.files.getlist('file')
        filename = rw.read_csvs(uploaded_files)
        all_data = dbm.mongo_to_df(filename)
        data = all_data['SentimentText']
        print(str(filename))
        vi.generate_workcloud(data, filename)
        return redirect(url_for('preprocessing', filename=filename))

    return render_template('upload.html', title='Upload')


@app.route('/preprocessing/<filename>', methods=['GET', 'POST'])
def preprocessing(filename):
    datos = [{'name':'ToLower'}, {'name':'DeleteAccent'},
     {'name':'Tokenize'}, {'name':'DeleteStopwords'}, {'name':'Stemmer'},
    {'name':'tfidf'}]
    alg = [{'name':'Bayes'}, {'name':'SVM'}]

    params_bayes = ['alpha', 'fit_prior', 'class_prior']
    paramns_svm = ['gamma', 'C', 'kernel', 'degree', 'coef0', 'shrinking', 'probability', 'tol', 'cache_size', 'class_weight', 'verbose', 'max_iter', 'decision_function_shape', 'random_state']
    
    if request.method == 'POST':
        options = request.form.getlist('preprocessing')
        algoritmo = request.form.get('algoritmo')
        cv = int(request.form.get('cv'))
        name = request.form.get('mn')
        lang = request.form.get('sel_lang')
        t_size = float(request.form.get('text_parti'))
        print('El lenguaje elegido es: ' + lang)
        print('La división elegida es: ' + str(t_size))
        if not options:
            print('No se ha seleccionado ningún algoritmo de preprocesamiento.')
            options = ['ToLower', 'DeleteAccent', 'Tokenize', 'DeleteStopwords',
            'Stemmer', 'tfidf']
            print('Al no seleccionar nada el usuario se aplicarán los siguientes:')
            for o in options:
                print(o)
        else:
            print('Se han seleccionado métodos para el preprocesamiento.')
        if algoritmo == None:
            print('No se ha seleccionado ningún algoritmo. ')
            print('Por tanto se aplicará Bayes.')
            algoritmo = 'Bayes'
        else:
            print('Se la seleccionado el algoritmo: ' + algoritmo)

        # Cogemos los datos
        data = dbm.mongo_to_df(filename)
        train, test = pr.split_to_train_test(data, t_size)
        # options = request.form.getlist('preprocessing')
        train_text = train["SentimentText"]
        train_class = train["Sentiment"]
        test_text = test["SentimentText"]
        checker_tfidf = None
        if not 'Tokenize' in options:
            if 'ToLower' in options and 'DeleteAccent' in options:
                options.insert(2,'Tokenize')
                print('Añado Tokenize en el 2')
            elif not 'ToLower' in options and not 'DeleteAccent' in options:
                options.insert(0,'Tokenize')
                print('Añado Tokenize en el 0')
            elif (not 'ToLower' in options and 'DeleteAccent' in options) or ('ToLower' in options and not 'DeleteAccent' in options):
                options.insert(1, 'Tokenize')
                print('Añado Tokenize en el 1')

        for o in options:
            if o == 'ToLower':
                print(o)
                train_text, test_text = pr.to_lower(train_text, test_text)
            if o == 'DeleteAccent':
                print(o)
                train_text, test_text = pr.delete_accent(train_text, test_text)
            if o == 'Tokenize':
                print(o)
                train_text, test_text = pr.tokenize(train_text, test_text)
            if o == 'DeleteStopwords':
                print(o)
                train_text, test_text = pr.delete_stopwords(train_text, test_text)
            if o == 'PosTage':
                print(o)
            if o =='Stemmer':
                print(o)
                train_text, test_text = pr.stemmer(train_text, test_text, lang)
            if o == 'tfidf':
                print(o)
                data_text, features = pr.tfidf(train_text, test_text)
                train_text = data_text['train']
                test_text = data_text['test']
                checker_tfidf = True

        if checker_tfidf == None:
            data_text, features = pr.tfidf(train_text, test_text)
            train_text = data_text['train']
            test_text = data_text['test']
            options.append('tfidf')
            print('No se seleccionó TF-IDF, pero se ejecutó después')
        print('Se entrenará el modelo:')
        if algoritmo == 'Bayes':
            alg = Bayes()
        else:
            alg = SVM()
        if cv > 1:
            print('Cross-Validation')
            cv_score = cross_validate(alg, train_text, train_class, cv=cv)
            print('Algoritmo entrenado')
            pred = cv_score['test_score']
        else:
            print('Hold-Out')
            alg.fit(train_text, train_class)
            print('Algoritmo entrenado')
            pred = alg.predict(test_text)
            print('Predicción: ' + str(pred))
        # TODO
        # Añadir resutador
        # Se guarda el modelo
        lang_part = {'Lang':lang, 'Part': t_size}
        details = dbm.save_model(model=alg, collection='PrModels', model_name=name,
                     preprocessing_lists=options, alg_used=algoritmo, features=features,
                     lang_part=lang_part, archives=filename)
        print(details['inserted_id'])
        id = details['inserted_id']
        return redirect(url_for('ver_modelo', id_modelo=id))
    # path = 'images/' + filename + '.png'
    path ='images/' + filename + '.png'
    return render_template('train.html', title=filename, data=datos, alg=alg, url=path)


"""
@app.route('/subida/<filename>')
def subida(filename):
    # display_html(filename, raw=true)
    df = dbm.mongo_to_df(filename)
    if (len(df) > 0):
        print(len(df))
        print(df)
        return render_template('subida.html', title=filename)
    else:
        return redirect(url_for('index'))


@app.route('/auxiliar/<filename>', methods=['GET', 'POST'])
def aux(filename):
    if request.method == 'POST':
        data = dbm.mongo_to_df(filename)
        train, test = pr.split_to_train_test(data)
        options = request.form.getlist('preprocessing')
        train_text = train["SentimentText"]
        train_class = train["Sentiment"]
        test_text = test["SentimentText"]
        for o in options:
            if o == 'ToLower':
                train_text, test_text = pr.to_lower(train_text, test_text)
            if o == 'Tokenize':
                train_text, test_text = pr.tokenize(train_text, test_text)
                print(o)
            if o == 'DeleteStopwords':
                train_text, test_text = pr.delete_stopwords(train_text, test_text)
            if o == 'PosTage':
                print(o)
                print('Mostrado')
            if o =='Stemmer':
                train_text, test_text = pr.stemmer(train_text, test_text)
                print(o)
            if o == 'DeleteAccent':
                print(o)
            if o == 'tfidf':
                train_text, test_text = pr.tfidf(train_text, test_text)
                print(o)
        # Guardar train y test y pasar a selección del algoritmo de clasificación
        data_id = dbm.save_train_test(train_text, train_class, test_text)
        # return render_template('subida.html', title=filename, data_id=data_id)
        return redirect(url_for('classification', filename=filename, data_id=data_id))
    datos = [{'name':'ToLower'}, {'name':'Tokenize'}, {'name':'PosTage'},
     {'name':'DeleteStopwords'}, {'name':'Stemmer'}, {'name':'DeleteAccent'},
    {'name':'tfidf'}]

    return render_template('prepro.html', title=filename,
        data = datos)

@app.route('/clasification/<filename>/<data_id>')
def classification(filename, data_id):
    return 0
"""
