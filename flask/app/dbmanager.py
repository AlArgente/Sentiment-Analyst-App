from pymongo import MongoClient
from flask_pymongo import PyMongo
import gridfs
import pandas as pd
import time
import pickle
from config import Config
import numpy as np
import json
import dill
from bson.objectid import ObjectId
from bson.binary import Binary

# Función para recuperar el dataframe de la base de datos
# Recibe el nombre de la colección donde tomar el DataFrame
# Devuelve el DataFrame en collection
def mongo_to_df(id):

    db = db_conection()
    pickle_data = db['dataTrain'].find( { '_id': ObjectId(id) } )
    json_data = {}
    for x in pickle_data:
        json_data = x
    df = pickle.loads(json_data['Data'])
    ###################
    # print('Prueba cargado de datos guardados como pickle')
    # mongo = MongoClient()
    # db = mongo.myDatabase
    # cursor = db['PruebaGuardado']
    # pickle_data = db['PruebaGuardado'].find( { '_id': ObjectId("5c86178c9ce5883fb058e580") } )
    # json_data = {}
    # for x in pickle_data:
    #    json_data = x
    # data = pickle.loads(json_data['Data'])
    # print(data)
    # print('Fin prueba cargado de datos guardados como pickle')
    ###################
    # df = pd.DataFrame(list(cursor))

    # del df['_id']

    return df

# Función para coenctar con la base de datos
# Devuelve una conexión a la base de datos
def db_conection():
    mongo = MongoClient()
    db = mongo.myDatabase
    return db

# Function to save a Machine Learning Model into the MongoDB
# @param, model, modelo de ML a guardar
# @param collection, colelction en la que se guardará el modelo
# @param model_name, nombre del modelo a guardar
# @param preprocessing_lists, list con los métodos usados para el preprocesamiento
# @param alg_used, algoritmo de clasificación usado para entrenar el modelo
# @ret, detalles del modelo guardado
def save_model(model, collection, model_name, preprocessing_lists, alg_used, features, lang_part, archives):
    # First, pickling the model
    pickle_model = pickle.dumps(model)

    # Pickling the features
    print('Pickling the Features')
    pickle_features = pickle.dumps(features)
    print('Pickling the Features 2')
    # Coneccting to the bd
    mongo = MongoClient()
    db = mongo.myDatabase

    # Load the collection:
    cursor = db[collection]
    info = cursor.insert_one({'model': pickle_model, 'created_time':time.time(),
                              'Prepro':preprocessing_lists, 'alg_used':alg_used,
                              'Features':pickle_features, 'hora':time.asctime( time.localtime(time.time())),
                              'nombre':model_name, 'lang': lang_part['Lang'], 't_size': lang_part['Part'],
                              'id_archivos':archives})

    # print(info)
    print(info.inserted_id, 'saved with this id succesfully!')

    details = {
        'inserted_id':info.inserted_id,
        'created_time':time.time()
    }
    print(time.asctime( time.localtime(time.time()) ))
    return details

# Función para obtener un modelo dado su id y una collection
# @param model_id, id del modelo a buscar
# @param collection, colleción en la que buscar el modelo
# @ret, modelo encontrado
def get_model_by_id(model_id, collection):
    mongo = MongoClient()
    db = mongo.myDatabase
    model1 = db[collection].find( { '_id': ObjectId(model_id) } )
    # model1 = db[collection].find_one()
    print(model_id)
    json_data = {}

    for x in model1:
        json_data = x

    pickle_model = json_data['model']
    model_data = {
        'Prepro':json_data['Prepro'],
        'Alg':json_data['alg_used'],
        'Features':pickle.loads(json_data['Features']),
        'Hora':json_data['hora'],
        'nombre':json_data['nombre'],
        'lang': json_data['lang'],
        't_size': json_data['t_size']
    }
    return pickle.loads(pickle_model), model_data

def get_pickle_model(model_id, collection):
    db = db_conection()
    model = db[collection].find( { '_id': ObjectId(model_id) } )
    json_data = {}
    for x in model:
        json_data = x

    pickle_model = json_data['model']
    model_name = json_data['nombre']
    print('Previo return de devolver el pickle model')
    return pickle_model, model_name

# Función para obtener todos los IDs de los modelos guardados en una collection
# @param collection, colección en la que se buscarán los modelos almacenados
# @ret, lista con todos los ids de los modelos
def get_all_models_saved(collection):
    db = db_conection()
    cursor = db[collection]
    # Guardamos toda la colección en una var
    all_models = db[collection].find()
    all_ids = []
    all_hours = []
    # Guardamos todos los ids
    for info in all_models:
        all_ids.append(info['nombre'])
        all_hours.append(info['_id'])
    return all_ids, all_hours

# Función para guardar en la base de datos los datos de train y test tras el
# preprocesamiento. Para ello guardará los datos preprocesados y las etiquetas
# de los datos de train
# @param train_text, datos de train preprocesados
# @param train_class, etiquetas de los datos de entrenamiento
# @param test_text, datos de test preprocesados
# @ret data_id, id de los datos para poder obtenerlos después
def save_train_test(train_text, train_class, test_text):
    print('Llego aquí')
    data = {'train_text':train_text, 'train_class':train_class, 'test_text':test_text}
    # data['test_text'] =  train_text.tostring()
    data['train_text'] = Binary(pickle.dumps(train_text, protocol=2), subtype=128)
    print('Paso el primero')
    # data['train_text'] =  test_text.tostring()
    data['test_text'] = Binary(pickle.dumps(test_text, protocol=2), subtype=128)
    print('Paso el segundo')
    # train_class = np.array(train_class)
    data['train_text'] = Binary(pickle.dumps(train_class, protocol=2), subtype=128)
    data['train_class'] = train_class
    print('Paso el tercero')
    print('Muestro data:')
    print(type(data['train_text']))
    print('Data mostrado')
    db = db_conection()
    fs = gridfs.GridFS(db, collection='preprocessedData')

    # data_aux = data['train_text'].tolist()
    # print('Antes de TRAIN TO JSON')
    # data['train_text'] = json.dumps(data_aux)
    # print('Antes de TRAIN CLASS TO JSON')
    # data_aux = data['train_class'].tolist()
    # data['train_class'] = json.dumps(data_aux)
    # print('Antes de TEST TO JSON')
    # data_aux = data['test_text'].tolist()
    # data['test_text'] = json.dumps(data_aux)
    print('Antes de guardarrr')
    json_data = json.dumps(data)
    data_id = fs.put(json_data, encoding='utf-8')
    return data_id

# FUnción para, a partir de un id, coger los datos de preprocesamiento que se usarán
# para el entrenamiento del modelo.
# @param data_id, id de los datos a coger
# @ret, datos
def get_data_by_id(data_id):
    db = db_conection()
    data = db['preprocessedData'].find( {'_id': ObjectId(data_id)} )
    return data


# Función para guardar los resultados del modelo.
# Se guardarán: recall, precision y F1-Score
def save_results_model(id_modelo, results):
    db = db_conection()
    cursor = db['Resultados']
    pickle_pred = pickle.dumps(results['Predict'])
    results['Predict'] = pickle_pred
    info = cursor.insert_one({'id_modelo': id_modelo, 'results':results})
    print(info.inserted_id, 'saved with this id succesfully!')


def get_saved_results_model(id_modelo):
    db = db_conection()
    resultados = db['Resultados'].find( {'id_modelo':id_modelo})
    for x in resultados:
        json_data = x

    resultados = json_data['results']
    resultados['Predict'] = pickle.loads(resultados['Predict'])
    # return json_data['results']
    return resultados

def save_archivos(data):
    db = db_conection()
    # db['dataTrain'].drop()
    # db['dataTrain'].insert(d_dict)
    # ret = "dataTrain"
    ########################
    # Convertimos data a pickle item
    d_pickle = pickle.dumps(data)
    cursor = db['dataTrain']
    # Lo guardamos en la BD
    info = cursor.insert_one({'Data':d_pickle})
    # Guardamos el ID de los datos para devolverlo y posteriormente
    # recuperar los datos
    ret = info.inserted_id
    print('Guardado como pickle data')
    print(str(info.inserted_id))
    return ret
