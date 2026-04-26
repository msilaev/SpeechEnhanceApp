import os
from config import Config
from flask import Flask
from flask_wtf.csrf import CSRFProtect
import onnxruntime as ort

app = Flask(__name__)
app.config.from_object(Config)

csrf = CSRFProtect(app)

os.makedirs(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), exist_ok=True)

# Load ONNX inference sessions once at startup to avoid per-request overhead
_model_files = {
    'upsample16_gan': 'upsample16_gan_500.onnx',
    'upsample48_gan': 'upsample48_gan_500.onnx',
    'upsample16_audiounet': 'upsample16_audiounet_500.onnx',
    'upsample48_audiounet': 'upsample48_audiounet_500.onnx',
}
app.config['ONNX_SESSIONS'] = {
    name: ort.InferenceSession(os.path.join(app.root_path, app.config['MODEL_FOLDER'], fname))
    for name, fname in _model_files.items()
}

from . import routes

