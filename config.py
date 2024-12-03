import os

class Config:
    UPLOAD_FOLDER = 'uploads'
    MODEL_FOLDER = 'models'
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'any_key'
    SAMPLING_RATE = 16000