from flask import Flask
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
# Here I add more variables as needed

from app import routes
