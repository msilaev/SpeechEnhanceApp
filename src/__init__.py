import os
from config import Config
from flask import Flask
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
app.config.from_object(Config)

csrf = CSRFProtect(app)

os.makedirs(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), exist_ok=True)

from . import routes

