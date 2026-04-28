from . import app
from flask import (render_template, request, url_for,
                   jsonify, redirect, send_from_directory)

from src.upsample48 import Upsample48
from src.upsample16 import Upsample16
from src.upsample16_audiounet import Upsample16AudioUnet
from src.upsample48_audiounet import Upsample48AudioUnet

import json
import os
import time
import joblib
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import librosa as lb
import soundfile as sf
import numpy as np

def get_spectrum(filename, sr,  n_fft=2048):

    audio_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
    
    x, sr = lb.load(audio_path, sr=sr)
        
    S = lb.stft(x, n_fft = n_fft)
    p = np.angle(S)

    S_dB = lb.amplitude_to_db(np.abs(S), ref=np.max)
    
    return S_dB

def save_spectrum(S, sr, hop_length, output_filename='spectrogram.png', type = "original"):
    
    plt.figure(figsize=(5, 5))  # Adjust the figure size for smaller paper size

    plt.rcParams.update({'font.size': 20})  # General font size
    plt.rcParams.update({'axes.titlesize': 20})  # Title font size
    plt.rcParams.update({'axes.labelsize': 20})  # X and Y label font size
    plt.rcParams.update({'legend.fontsize': 20})  # Legend font size
    plt.rcParams.update({'xtick.labelsize': 20})  # X tick label font size
    plt.rcParams.update({'ytick.labelsize': 20})  # Y tick label font size

    lb.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.yticks(ticks=np.arange(0, sr // 2 + 1, sr//8),
               labels=[f'{x / 1000:.1f}' for x in np.arange(0, sr // 2 + 1, sr//8)])



    max_time = S.shape[1] * hop_length / sr  # Convert frames to seconds
    plt.xlim(0, max_time)  # Limit x-axis to the range of the data
    plt.xticks(ticks=[i for i in range(int(max_time) + 1) if i <= max_time])

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (kHz)')

    plt.tick_params(axis='both', which='major')  # Major ticks
    plt.tick_params(axis='both', which='minor')  # Minor ticks

    plt.tight_layout()

    output_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], output_filename)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # High DPI for better quality
    plt.close()

    return output_path

# Minimum acceptable native sample rate — guards against truly unusable input.
# Lower-SR files are resampled by librosa before processing.
_EXPECTED_SR = {
    'upsample16_gan':       4000,
    'upsample48_gan':       4000,
    'upsample16_audiounet': 4000,
    'upsample48_audiounet': 4000,
}


def get_mel_energy(filename, sr, n_fft, hop_length, n_mels=256):
    audio_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
    y, _ = lb.load(audio_path, sr=sr, mono=True)
    mel = lb.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                    n_mels=n_mels, fmax=sr / 2, htk=True)
    log_mel = lb.power_to_db(mel, ref=1.0)
    return log_mel.mean(axis=1)  # time-averaged, shape (n_mels,)


def _segment_audio(y, window_size=16384, stride=16384):
    """Split signal into non-overlapping windows, zero-padding the last one if needed."""
    n = len(y)
    if n < window_size:
        pad = window_size - n
    else:
        remainder = (n - window_size) % stride
        pad = 0 if remainder == 0 else stride - remainder
    y = np.pad(y, (0, pad), mode='constant')
    return np.stack([y[i:i + window_size]
                     for i in range(0, len(y) - window_size + 1, stride)])


def _window_mel(window, sr, n_fft, hop_length, n_mels=256):
    """Compute time-averaged log-mel vector for a single audio window (htk scale)."""
    mel = lb.feature.melspectrogram(y=window, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                    n_mels=n_mels, fmax=sr / 2, htk=True)
    return lb.power_to_db(mel, ref=1.0).mean(axis=1)  # (n_mels,)


def save_mel_energy_combined(entries, sr, n_mels=256, output_filename='mel_energy_combined.png'):
    """Plot multiple mel energy curves on one axes.

    entries: list of (label, color, log_mel_vec)
    """
    freqs = lb.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2, htk=True)

    plt.figure(figsize=(8, 4))
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'legend.fontsize': 12})

    for label, color, vec in entries:
        plt.plot(freqs / 1000, vec, color=color, linewidth=1.5, label=label)

    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Energy (dB)')
    plt.xlim(0, sr / 2000)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def cleanup_uploads():
    """Remove per-request temp files from the uploads folder."""
    for fname in ['original.wav', 'user.wav', 'processed.wav',
                  'spectrogram_uploaded.png', 'spectrogram_original.png', 'spectrogram_processed.png',
                  'mel_energy_combined.png', 'classification.json']:
        path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], fname)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def process_audio(filename, model_type, output_filename, target_sr):

    try:
        audio_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)

        # Validate that the file exists before attempting to process it
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Uploaded file not found: {filename}")

        # Validate native sample rate without loading the full audio
        file_info = sf.info(audio_path)
        native_sr = file_info.samplerate
        min_sr = _EXPECTED_SR[model_type]
        if native_sr < min_sr:
            raise ValueError(
                f"File sample rate is {native_sr} Hz, but this model requires "
                f"at least {min_sr} Hz audio. Please upload a higher-quality file."
            )

        sessions = app.config['ONNX_SESSIONS']

        # Always load at the model's full SR — librosa resamples lower-SR files
        # automatically. The full decimate+spline pipeline always runs so the model
        # receives the same type of input it was trained on regardless of source SR.
        if model_type in ('upsample48_gan', 'upsample48_audiounet'):
            load_sr = 48000
        else:
            load_sr = app.config['SAMPLING_RATE']

        y, sr = lb.load(audio_path, sr=load_sr)

        if model_type == 'upsample48_gan':
            processor = Upsample48(y)
            session = sessions['upsample48_gan']
        elif model_type == 'upsample16_gan':
            processor = Upsample16(y)
            session = sessions['upsample16_gan']
        elif model_type == 'upsample16_audiounet':
            processor = Upsample16AudioUnet(y)
            session = sessions['upsample16_audiounet']
        elif model_type == 'upsample48_audiounet':
            processor = Upsample48AudioUnet(y)
            session = sessions['upsample48_audiounet']
        else:
            raise ValueError("Invalid model specified.")

        # Save the original upload (resampled to target_sr) before processing
        sf.write(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], 'original.wav'), y, target_sr)

        try:
            processed_audio, input_audio, processing_time, duration = processor.predict(session)
        except Exception as a:
            raise RuntimeError(f"processor.predict failed: {str(a)}")

        sf.write(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], output_filename), processed_audio, target_sr)
        sf.write(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], 'user.wav'), input_audio, target_sr)

        return processing_time, duration
    
    except Exception as e:

        raise RuntimeError(f"Audio processing failed: {str(e)}")

def classify_audio(filename, sr, n_fft, hop_length, classifier, window_size=16384):
    """Classify audio as real/enhanced using the same windowed mel pipeline as
    src_classification_16: chunk → per-window mel embedding → LDA → aggregate."""
    if classifier is None:
        return None

    audio_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
    y, _ = lb.load(audio_path, sr=sr, mono=True)

    windows = _segment_audio(y, window_size=window_size)
    embeddings = np.array([_window_mel(w, sr, n_fft, hop_length) for w in windows])

    X = classifier['scaler'].transform(embeddings)
    preds = classifier['model'].predict(X).astype(float)
    proba = classifier['model'].predict_proba(X)
    pos_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

    mean_p_real = float(np.mean(pos_prob))
    majority = int(np.round(np.mean(preds)))
    label = classifier['label_map'].get(majority, str(majority))

    return {'label': label, 'p_real': round(mean_p_real, 4), 'n_windows': int(len(windows))}


def save_classification(result):
    path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], 'classification.json')
    with open(path, 'w') as f:
        json.dump(result or {}, f)


def _visuals_48(audio_file, spec_out):
    save_spectrum(get_spectrum(audio_file, sr=48000, n_fft=3*2048),
                  sr=48000, hop_length=3*2048 // 4, output_filename=spec_out)


def _visuals_16(audio_file, spec_out):
    save_spectrum(get_spectrum(audio_file, sr=16000, n_fft=2048),
                  sr=16000, hop_length=2048 // 4, output_filename=spec_out)


def _mel_combined_48():
    entries = [
        ('Uploaded',     '#1f77b4', get_mel_energy('original.wav', sr=48000, n_fft=4096, hop_length=512)),
        ('Model Input',  '#ff7f0e', get_mel_energy('user.wav',     sr=48000, n_fft=4096, hop_length=512)),
        ('Model Output', '#2ca02c', get_mel_energy('processed.wav',sr=48000, n_fft=4096, hop_length=512)),
    ]
    save_mel_energy_combined(entries, sr=48000)


def _mel_combined_16():
    entries = [
        ('Uploaded',     '#1f77b4', get_mel_energy('original.wav', sr=16000, n_fft=4096, hop_length=256)),
        ('Model Input',  '#ff7f0e', get_mel_energy('user.wav',     sr=16000, n_fft=4096, hop_length=256)),
        ('Model Output', '#2ca02c', get_mel_energy('processed.wav',sr=16000, n_fft=4096, hop_length=256)),
    ]
    save_mel_energy_combined(entries, sr=16000)


@app.route('/', methods=['GET', 'POST'])
def file_upload():

    if request.method == 'POST':

        cleanup_uploads()

        file = request.files.get('file')

        if not file or file.filename == '' or (not (file.filename.endswith('.wav') or file.filename.endswith('.flac'))):
            return jsonify({'error': 'Please upload a WAV or FLAC file'})

        max_file_size = 50 * 1024 * 1024

        try:
            filename = secure_filename(file.filename)

            if filename != '':
            
                file_ext = os.path.splitext(filename)[1]

                file.save(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], "user.wav"))

                file.close()

                audio = os.path.join(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], "user.wav"))

                file_size = os.path.getsize(audio)

                if (file_size > max_file_size):
                    return jsonify({'error': "File size exceeds the limit of 50 MB!"})
            
            return redirect(url_for('upload_complete', result = "Upload complete" ))
           
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('upload.html')


@app.route('/upload_complete/<result>')
def upload_complete(result):   
   return render_template("upload_complete.html", result = result)


@app.route('/upsample48')
def upsample48(filename='user.wav'):
    try:
        processing_time, duration = process_audio(filename, 'upsample48_gan', 'processed.wav', 48000)
        _visuals_48('original.wav',  'spectrogram_uploaded.png')
        _visuals_48('user.wav',      'spectrogram_original.png')
        _visuals_48('processed.wav', 'spectrogram_processed.png')
        _mel_combined_48()
        save_classification(classify_audio('processed.wav', 48000, 4096, 512,
                                           app.config['CLASSIFIERS'].get('upsample48_gan')))
        return redirect(url_for('report', result=f'Upsampling complete! Processing time {processing_time:.2f} s, '
                                                 f'duration {duration:.2f} s'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upsample16')
def upsample16(filename='user.wav'):
    try:
        processing_time, duration = process_audio(filename, 'upsample16_gan', 'processed.wav', 16000)
        _visuals_16('original.wav',  'spectrogram_uploaded.png')
        _visuals_16('user.wav',      'spectrogram_original.png')
        _visuals_16('processed.wav', 'spectrogram_processed.png')
        _mel_combined_16()
        save_classification(classify_audio('processed.wav', 16000, 4096, 256,
                                           app.config['CLASSIFIERS'].get('upsample16_gan')))
        return redirect(url_for('report', result=f'Upsampling complete! Processing time {processing_time:.2f} s, '
                                                 f'duration {duration:.2f} s'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upsample16_audiounet')
def upsample16_audiounet(filename='user.wav'):
    try:
        processing_time, duration = process_audio(filename, 'upsample16_audiounet', 'processed.wav', 16000)
        _visuals_16('original.wav',  'spectrogram_uploaded.png')
        _visuals_16('user.wav',      'spectrogram_original.png')
        _visuals_16('processed.wav', 'spectrogram_processed.png')
        _mel_combined_16()
        save_classification(classify_audio('processed.wav', 16000, 4096, 256,
                                           app.config['CLASSIFIERS'].get('upsample16_audiounet')))
        return redirect(url_for('report', result=f'Upsampling complete! Processing time {processing_time:.2f} s, '
                                                 f'duration {duration:.2f} s'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upsample48_audiounet')
def upsample48_audiounet(filename='user.wav'):
    try:
        processing_time, duration = process_audio(filename, 'upsample48_audiounet', 'processed.wav', 48000)
        _visuals_48('original.wav',  'spectrogram_uploaded.png')
        _visuals_48('user.wav',      'spectrogram_original.png')
        _visuals_48('processed.wav', 'spectrogram_processed.png')
        _mel_combined_48()
        save_classification(classify_audio('processed.wav', 48000, 4096, 512,
                                           app.config['CLASSIFIERS'].get('upsample48_audiounet')))
        return redirect(url_for('report', result=f'Upsampling complete! Processing time {processing_time:.2f} s, '
                                                 f'duration {duration:.2f} s'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/report/<result>')
def report(result):
    classif_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], 'classification.json')
    classification = None
    if os.path.exists(classif_path):
        with open(classif_path) as f:
            classification = json.load(f) or None
    return render_template("report.html", result=result, cache_bust=int(time.time()),
                           classification=classification)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)