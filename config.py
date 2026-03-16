import os

class Config:
    UPLOAD_FOLDER = 'uploads'
    MODEL_FOLDER = 'models'
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY environment variable not set")
    SAMPLING_RATE = 16000