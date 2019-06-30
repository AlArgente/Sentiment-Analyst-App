import pandas as pd
import csv, os
from app import app
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from app import dbmanager as dbm
import pickle

# Function to check if the uploaded file is allowed in our up.
# Función que comprueba si el archivo que se ha subido está entre los
# permitidos en nuestra app.
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in app.config['ALLOWD_EXTENSIONS']

# Read multiple csv file uploaded by the user
# Lee varios archivos csv subidos por el usuario
# Devuelve el nombre de la colección donde se ha almacenado el DataFrame
def read_csvs(filename):
    df = []
    archivo = ''
    if len(filename) > 0:
        for file in filename:
            archivo = file.filename
            if allowed_file(archivo):
                name = secure_filename(archivo)
                df_aux = pd.read_csv(file, index_col=None, header=0, encoding = "ISO-8859-1")
                df.append(df_aux)
                namedata = archivo
        data = pd.concat(df, axis = 0, ignore_index = True)
        # Pasamos todo el DataFrame a Json para guardarlo en la BD
        # d_dict = data.to_dict(orient='records')
        # Lo almacenamos en la base de datos.
        ret = dbm.save_archivos(data)
        """
        # Accedemos a la Base de Datos
        mongo = MongoClient()
        db = mongo.myDatabase
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
        """
        #######################
    return ret


# AÑADIR FUNCIÓN PARA ESCRIBIR EN UN CSV
# Se usará mongo_to_df de preprocessing.py para tomar de la BD el modelo,
# para convertirlo en un DataFrame, y posteriormente se mandará como un pickle,
# al cliente.

def write_csv(id_model):
    resultados = dbm.get_saved_results_model(id_modelo)
    # return send_file(model, mimetype='pickle/pickle')
    tmp_name = id_modelo + '.csv'
    tmp_path = 'app/tmpfiles/tmp.csv'
    df.to_csv('app/tmpfiles/tmp.csv')
    return resultados, tmp_path
