import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from pymongo import MongoClient
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from stop_words import get_stop_words
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# Clase de preprocesamiento
class Preprocessing():
    # Función para recuperar el dataframe de la base de datos
    # Recibe el nombre de la colección donde tomar el DataFrame
    def mongo_to_df(collection):
        mongo = MongoClient()
        db = mongo.myDatabase

        cursor = db[collection].find({})

        df = pd.DataFrame(list(cursor))

        del df['_id']

        return df

    # Función que divide el dataset en train y test
    def split_to_train_test(data):
        train, test = train_test_split(data, test_size=0.3, random_state=42)
        return train, test

    # Función para Tokenizar el texto
    def tokenize(train, test):
        train_token = []
        test_token = []
        for w in train:
            train_token.append(word_tokenize(str(w)))
        for w in test:
            test_token.append(word_tokenize(str(w)))
        return train_token, test_token

    # Funcino para eliminar las stopwords del train-test
    def delete_stopwords(train, test):
        # Recopilamos todas las stopwords en español y en inglés
        stop_words_es = list(get_stop_words('spanish'))
        nltk_words_es = list(stopwords.words('spanish'))
        stop_words_en = list(get_stop_words('en'))
        nltk_words_en = list(stopwords.words('english'))
        # Juntamos todas las stopwords en una
        all_stop_words = stop_words_en + stop_words_es + nltk_words_en + nltk_words_es

        # Eliminamos las stopwords del train
        train_ds = []
        for word in train:
            for w in word:
                if w not in all_stop_words:
                    train_ds.append(w)

        # Eliminamos las stopwords del test
        test_ds = []
        for word in test:
            for w in word:
                if w not in all_stop_words:
                    test_ds.append(w)
        return train_ds, test_ds

    # Función para etiquetar gramaticalmente las palabras
    def pos_tag(train, test):
        pt_train = nltk.pos_tag(train)
        pt_test = nltk.pos_tag(test)
        return pt_train, pt_test

    # Función para calcular la raíz de las palabras que les pasemos
    def stemmer(train,test):
        # Cargamos el stemmer
        ps = SnowballStemmer("english")
        train_stemmer = []
        test_stemmer = []
        for sentence in train:
            for token in sentence:
                train_stemmer.append(ps.stem(token))
        for sentence in test:
            for token in sentence:
                test_stemmer.append(ps.stem(token))
        return train_stemmer, test_stemmer

    def delete_accent(train, test):
        return "a"

    # Función para convertir todas las letras a minúsculas
    def to_lower(train, test):
        tr_tl = tr.lower()
        te_tl = te.lower()
        return tr_tl, te_tl

    # Función para calcular el Tfidf.
    def tfidf(train, test):
        vectorizer = TfidfVectorizer()

        return train, test
