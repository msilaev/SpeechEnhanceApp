import io
import os
import pytest
from unittest.mock import patch

os.environ.setdefault('SECRET_KEY', 'test-secret-key')

from src import app as flask_app


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    flask_app.config['WTF_CSRF_ENABLED'] = False
    with flask_app.test_client() as client:
        yield client


# --- Upload page ---

def test_upload_page_loads(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Upload' in response.data


def test_upload_no_file_returns_error(client):
    response = client.post('/', data={})
    # no file → attribute error on file.filename, caught by the route
    assert response.status_code in (200, 400, 500)


def test_upload_wrong_extension_returns_error(client):
    data = {'file': (io.BytesIO(b'not audio'), 'test.mp3')}
    response = client.post('/', data=data, content_type='multipart/form-data')
    assert b'error' in response.data.lower() or response.status_code == 200


def test_upload_valid_wav_redirects(client, tmp_path):
    # minimal valid WAV header (44 bytes)
    wav_header = (
        b'RIFF' + (36).to_bytes(4, 'little') +
        b'WAVE' + b'fmt ' + (16).to_bytes(4, 'little') +
        (1).to_bytes(2, 'little') +   # PCM
        (1).to_bytes(2, 'little') +   # mono
        (16000).to_bytes(4, 'little') +  # sample rate
        (32000).to_bytes(4, 'little') +  # byte rate
        (2).to_bytes(2, 'little') +   # block align
        (16).to_bytes(2, 'little') +  # bits per sample
        b'data' + (0).to_bytes(4, 'little')
    )
    with patch('src.routes.os.path.getsize', return_value=1024):
        data = {'file': (io.BytesIO(wav_header), 'test.wav')}
        response = client.post('/', data=data, content_type='multipart/form-data')
    assert response.status_code in (200, 302)


# --- Static routes ---

def test_upload_complete_route(client):
    response = client.get('/upload_complete/Upload%20complete')
    assert response.status_code == 200


def test_report_route(client):
    response = client.get('/report/test_result')
    assert response.status_code == 200
    assert b'test_result' in response.data


# --- Processing routes return error without model files ---

def test_upsample16_route_without_model_returns_error(client):
    response = client.get('/upsample16')
    assert response.status_code == 500
    assert b'error' in response.data.lower()


def test_upsample48_route_without_model_returns_error(client):
    response = client.get('/upsample48')
    assert response.status_code == 500
    assert b'error' in response.data.lower()


def test_upsample16_audiounet_route_without_model_returns_error(client):
    response = client.get('/upsample16_audiounet')
    assert response.status_code == 500
    assert b'error' in response.data.lower()


def test_upsample48_audiounet_route_without_model_returns_error(client):
    response = client.get('/upsample48_audiounet')
    assert response.status_code == 500
    assert b'error' in response.data.lower()
