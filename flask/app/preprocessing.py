import pandas as pd
import nltk
import numpy as np
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from pymongo import MongoClient
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from stop_words import get_stop_words
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode
from nltk.tokenize.treebank import TreebankWordTokenizer
"""
# Función para recuperar el dataframe de la base de datos
# Recibe el nombre de la colección donde tomar el DataFrame
def mongo_to_df(collection):
    mongo = MongoClient()
    db = mongo.myDatabase

    cursor = db[collection].find({})

    df = pd.DataFrame(list(cursor))

    del df['_id']

    return df
"""
# Función que divide el dataset en train y test, por defecto 80-20
def split_to_train_test(data, t_size=0.2):
    train, test = train_test_split(data, test_size=t_size, random_state=42)
    return train, test

# Función para tokenizar
# Por defecto se calculará para inglés, en un futuro se añadirá para español.
def word_tokenize_t(text, language='english', preserve_line=False):
    """
    Return a tokenized copy of *text*,
    using NLTK's recommended word tokenizer
    (currently an improved :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into words
    :type text: str
    :param language: the model name in the Punkt corpus
    :type language: str
    :param preserve_line: An option to keep the preserve the sentence and not sentence tokenize it.
    :type preserver_line: bool
    """
    _treebank_word_tokenizer = TreebankWordTokenizer()
    sentences = [text] if preserve_line else sent_tokenize(text, language)
    return [token for sent in sentences
            for token in _treebank_word_tokenizer.tokenize(sent)]

# Función para Tokenizar el texto
def tokenize(train, test, language='english'):
    # Tokenizamos primero en sentencias
    # tokenizer = nltk.data.load(‘tokenizers/punkt/english.pickle’)
    train_token = []
    test_token = []

    for instance in train:
        train_token.append(word_tokenize_t(instance, language))
    for instance in test:
        test_token.append(word_tokenize_t(instance, language))

    return train_token, test_token


# Funcino para eliminar las stopwords del train-test
def delete_stopwords(train, test, lang="english"):
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
        train_ds.append([w for w in word if w not in all_stop_words])

    # Eliminamos las stopwords del test
    test_ds = []
    for word in test:
        test_ds.append([w for w in word if w not in all_stop_words])
    return train_ds, test_ds

# Función para etiquetar gramaticalmente las palabras
def pos_tag(train, test):
    pt_train = nltk.pos_tag(train)
    pt_test = nltk.pos_tag(test)
    return pt_train, pt_test

# Función para calcular la raíz de las palabras que les pasemos
def stemmer(train,test, lang="english"):
    # Cargamos el stemmer
    stemmer = SnowballStemmer(lang)
    train_stemmer = []
    test_stemmer = []
    train_stemmer = [[stemmer.stem(token) for token in sentence] for sentence in train]
    test_stemmer = [[stemmer.stem(token) for token in sentence] for sentence in test]
    return train_stemmer, test_stemmer

# Función para eliminar acentos de las frases y convertir a unicode
def delete_accent(train, test):
    tr_da = []
    te_da = []

    for instance in train:
        tr_da.append(unidecode(instance))
    for instance in test:
        te_da.append(unidecode(instance))

    return tr_da, te_da

# Función para convertir todas las letras a minúsculas
def to_lower(train, test):
    tr_tl = train.str.lower()
    te_tl = test.str.lower()
    return tr_tl, te_tl

# Función para calcular el Tfidf.
def tfidf(train, test):
    print(train[:10][:])
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=lambda x: x, preprocessor=lambda x: x,
            token_pattern=None, lowercase=None, dtype='float')
    train_aux = np.array(train)
    test_aux = np.array(test)
    train_tfidf = vectorizer.fit_transform(train_aux)
    # feature_array = np.array(vectorizer.get_feature_names())
    vocabulary = vectorizer.vocabulary_
    print(train_tfidf.shape)
    # print(vectorizer.get_feature_names()[:20])
    test_tfidf = vectorizer.transform(test_aux)
    train_aux1 = train_tfidf.toarray()
    test_aux1 = test_tfidf.toarray()
    print('Train shape: '+ str(train_aux1.shape))
    print('Test shape: '+ str(test_aux1.shape))

    # Cambiar código para devolver esto, y así poder guardar la lista de las
    # características, para así poder usarlas posteriormente para visulización

    # test_tfidf = vectorizer.transform(test_aux)
    # train_aux1 = train_tfidf.toarray()
    # test_aux1 = test_tfidf.toarray()
    data = {'train':None, 'test':None}
    data = {'train':train_aux1, 'test':test_aux1}

    return data, vocabulary

    # return train_aux1, test_aux1

# Función para calcular el tfidf en la fase de producción.
# @param data, los datos sobre los que calcular el TF-IDF
# @param vectorizer, el TfidfVectorizer utilizado para entrenar al modelo,
# ya que sin este no se podrá calcular.
# @return data_tfidf, devuelve el TF_IDF de data
def tfidf_devolopment(data, vocabulary):
    """Función que calcula el TF-IDF sobre un nuevo conjunto de datos

    Devuelve el TF-IDF sobre el nuevo conjunto de datos

    Parámetros:
    data -- Conjunto de datos sobre el que clasificar el TF-IDF
    vectorizer -- TfidfVectorizer usado en el entrenamiento del modelo
    """
    vectorizer1 = TfidfVectorizer(analyzer="word", tokenizer=lambda x: x, preprocessor=lambda x: x,
            token_pattern=None, lowercase=None, dtype='float', vocabulary=vocabulary)
    # data_aux = np.array(data)
    #if (isinstance(vectorizer, TfidfVectorizer)):
    data_tfidf = vectorizer1.fit_transform(np.array(data))
    data_tfidf = data_tfidf.toarray()
    return data_tfidf
    """
    else:
        raise Exception('Vectorizer no es una instancia de TfidfVectorizer')
        return None
    """
