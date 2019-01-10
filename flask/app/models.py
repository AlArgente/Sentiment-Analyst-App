from app import mongo

class File(mongo.Document):
    created_at = mongo.DateTimeField(default=datetime.now())
    name = mongo.StringField()
    _file = mongo.FileField()
