import os
import warnings
from config import Config
from flask import Flask
from flask_wtf.csrf import CSRFProtect
import joblib
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
def _load_session(path):
    try:
        return ort.InferenceSession(path)
    except Exception:
        return None

app.config['ONNX_SESSIONS'] = {
    name: _load_session(os.path.join(app.root_path, app.config['MODEL_FOLDER'], fname))
    for name, fname in _model_files.items()
}

_classifier_files = {
    'upsample16_gan':       'stft_gan_16000_classifier.joblib',
    'upsample48_gan':       'stft_gan_48000_classifier.joblib',
    'upsample16_audiounet': 'stft_audiounet_16000_classifier.joblib',
    'upsample48_audiounet': 'stft_audiounet_48000_classifier.joblib',
}

def _load_classifier(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return joblib.load(path)
    except Exception:
        return None

app.config['CLASSIFIERS'] = {
    name: _load_classifier(os.path.join(app.root_path, app.config['MODEL_FOLDER'], fname))
    for name, fname in _classifier_files.items()
}

from . import routes

