from flask import Flask
from config import Config
from flask_pymongo import PyMongo
from flask_migrate import Migrate
# from flask.ext.mongoalchemy import MongoAlchemy

app = Flask(__name__)
app.config.from_object(Config)
# db = MongoAlchemy(app)
mongo = PyMongo(app)
migrate = Migrate(app,mongo)

from app import routes
