from flask import Flask
from config import Config
from flask_pymongo import PyMongo
from flask_migrate import Migrate
from flask_bootstrap import Bootstrap

app = Flask(__name__)
app.config.from_object(Config)
mongo = PyMongo(app)
bootstrap = Bootstrap(app)

from app import routes
