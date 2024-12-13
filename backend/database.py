from pymongo import MongoClient
from datetime import datetime

class Database:
    def __init__(self):
        # Connect to MongoDB (make sure MongoDB Compass is running)
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['student_persistence']
        
    def save_prediction(self, input_data, prediction):
        collection = self.db['predictions']
        record = {
            'timestamp': datetime.now(),
            'input': input_data,
            'prediction': prediction
        }
        return collection.insert_one(record)

    def get_predictions(self):
        collection = self.db['predictions']
        return list(collection.find({}, {'_id': 0}).sort('timestamp', -1).limit(10))
