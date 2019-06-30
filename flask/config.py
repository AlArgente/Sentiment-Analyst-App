import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'wubba-lubba-dub-dub'
    APP_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '/home/alberto/Desktop/flask/app/')
    MONGO_DBNAME = 'myDatabase'
    MONGO_URI = "mongodb://127.0.0.1:27017/myDatabase"
    ALLOWD_EXTENSIONS = set(['csv'])
    ALG_USED = ["Bayes", "SVM"]
