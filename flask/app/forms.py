from datetime import datetime
from flask_wtf import FlaskForm
from wtforms import StringField, MultipleFileField, SubmitField, TextAreaField

class UploadForm(Document):
    name = StringField()
    document = FileField()
