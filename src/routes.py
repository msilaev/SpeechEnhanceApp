from . import app
from flask import (render_template, request, url_for,
                   jsonify, redirect, send_from_directory)

from src.upsample48 import Upsample48
from src.upsample16 import Upsample16
from src.upsample16_audiounet import Upsample16AudioUnet
from src.upsample48_audiounet import Upsample48AudioUnet

import os
import time
import joblib
from werkzeug.utils import secure_filename
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

    cbar = plt.colorbar(format='%+2.0f dB')
   
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

# Expected minimum sample rate for each model type (used for validation)
_EXPECTED_SR = {
    'upsample16_gan':       16000,
    'upsample48_gan':       48000,
    'upsample16_audiounet': 16000,
    'upsample48_audiounet': 48000,
}


def get_mel_energy(filename, sr, n_fft, hop_length, n_mels=256):
    audio_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
    y, _ = lb.load(audio_path, sr=sr, mono=True)
    mel = lb.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                    n_mels=n_mels, fmax=sr / 2)
    log_mel = lb.power_to_db(mel, ref=1.0)
    return log_mel.mean(axis=1)  # time-averaged, shape (n_mels,)


def save_mel_energy(log_mel_vec, sr, n_mels=256, output_filename='mel_energy.png'):
    freqs = lb.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2, htk=True)

    plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 20, 'axes.titlesize': 20, 'axes.labelsize': 20,
                         'xtick.labelsize': 20, 'ytick.labelsize': 20})

    plt.plot(freqs / 1000, log_mel_vec, color='steelblue', linewidth=1.5)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Energy (dB)')
    plt.xlim(0, sr / 2000)
    plt.tight_layout()

    output_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def cleanup_uploads():
    """Remove per-request temp files from the uploads folder."""
    for fname in ['user.wav', 'processed.wav',
                  'spectrogram_original.png', 'spectrogram_processed.png',
                  'mel_energy_original.png', 'mel_energy_processed.png']:
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

        if model_type == 'upsample48_gan':
            y, sr = lb.load(audio_path, sr=48000)
            processor = Upsample48(y)
            session = sessions['upsample48_gan']

        elif model_type == 'upsample16_gan':
            y, sr = lb.load(audio_path, sr=app.config['SAMPLING_RATE'])
            processor = Upsample16(y)
            session = sessions['upsample16_gan']

        elif model_type == 'upsample16_audiounet':
            y, sr = lb.load(audio_path, sr=app.config['SAMPLING_RATE'])
            processor = Upsample16AudioUnet(y)
            session = sessions['upsample16_audiounet']

        elif model_type == 'upsample48_audiounet':
            y, sr = lb.load(audio_path, sr=48000)
            processor = Upsample48AudioUnet(y)
            session = sessions['upsample48_audiounet']

        else:
            raise ValueError("Invalid model specified.")

        try:
            processed_audio, input_audio, processing_time, duration = processor.predict(session)
        except Exception as a:
            raise RuntimeError(f"processor.predict failed: {str(a)}")

        output_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], output_filename)
        
        sf.write(output_path, processed_audio, target_sr)

        output_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], "user.wav")
        
        sf.write(output_path, input_audio, target_sr)
        
        return output_path, processing_time, duration
    
    except Exception as e:

        raise RuntimeError(f"Audio processing failed: {str(e)}")

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
def upsample48(filename = "user.wav"):
    
    try:
        
        processed_file, processing_time, duration = process_audio(filename, 'upsample48_gan', 'processed.wav', 48000)
        
        save_spectrum(get_spectrum("user.wav", sr=48000, n_fft=3*2048),
                      sr=48000, hop_length=3*2048 // 4,
                      output_filename='spectrogram_original.png', type='original')
        save_spectrum(get_spectrum("processed.wav", sr=48000, n_fft=3*2048),
                      sr=48000, hop_length=3*2048 // 4,
                      output_filename='spectrogram_processed.png', type='original')
        save_mel_energy(get_mel_energy("user.wav", sr=48000, n_fft=4096, hop_length=512),
                        sr=48000, output_filename='mel_energy_original.png')
        save_mel_energy(get_mel_energy("processed.wav", sr=48000, n_fft=4096, hop_length=512),
                        sr=48000, output_filename='mel_energy_processed.png')

        return redirect(url_for('report',
                                result=f'Upsampling complete! Processing time {processing_time:.2f} s, '
                                       f'duration {duration:.2f} s'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upsample16')
def upsample16(filename = "user.wav"):

    try:

        processed_file, processing_time, duration = process_audio(filename, 'upsample16_gan', 'processed.wav', 16000)

        save_spectrum(get_spectrum("user.wav", sr=16000, n_fft=2048),
                      sr=16000, hop_length=2048 // 4,
                      output_filename='spectrogram_original.png', type='original')
        save_spectrum(get_spectrum("processed.wav", sr=16000, n_fft=2048),
                      sr=16000, hop_length=2048 // 4,
                      output_filename='spectrogram_processed.png', type='original')
        save_mel_energy(get_mel_energy("user.wav", sr=16000, n_fft=4096, hop_length=256),
                        sr=16000, output_filename='mel_energy_original.png')
        save_mel_energy(get_mel_energy("processed.wav", sr=16000, n_fft=4096, hop_length=256),
                        sr=16000, output_filename='mel_energy_processed.png')

        return redirect(url_for('report', result=f'Upsampling complete! Processing time {processing_time:.2f} s, '
                                                 f'duration {duration:.2f} s'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upsample16_audiounet')
def upsample16_audiounet(filename="user.wav"):

    try:

        processed_file, processing_time, duration = process_audio(filename, 'upsample16_audiounet', 'processed.wav', 16000)

        save_spectrum(get_spectrum("user.wav", sr=16000, n_fft=2048),
                      sr=16000, hop_length=2048 // 4,
                      output_filename='spectrogram_original.png', type='original')
        save_spectrum(get_spectrum("processed.wav", sr=16000, n_fft=2048),
                      sr=16000, hop_length=2048 // 4,
                      output_filename='spectrogram_processed.png', type='original')
        save_mel_energy(get_mel_energy("user.wav", sr=16000, n_fft=4096, hop_length=256),
                        sr=16000, output_filename='mel_energy_original.png')
        save_mel_energy(get_mel_energy("processed.wav", sr=16000, n_fft=4096, hop_length=256),
                        sr=16000, output_filename='mel_energy_processed.png')

        return redirect(url_for('report', result=f'Upsampling complete! Processing time {processing_time:.2f} s, '
                                                 f'duration {duration:.2f} s'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upsample48_audiounet')
def upsample48_audiounet(filename="user.wav"):

    try:

        processed_file, processing_time, duration = process_audio(filename, 'upsample48_audiounet', 'processed.wav', 48000)

        save_spectrum(get_spectrum("user.wav", sr=48000, n_fft=3 * 2048),
                      sr=48000, hop_length=3 * 2048 // 4,
                      output_filename='spectrogram_original.png', type='original')
        save_spectrum(get_spectrum("processed.wav", sr=48000, n_fft=3 * 2048),
                      sr=48000, hop_length=3 * 2048 // 4,
                      output_filename='spectrogram_processed.png', type='original')
        save_mel_energy(get_mel_energy("user.wav", sr=48000, n_fft=4096, hop_length=512),
                        sr=48000, output_filename='mel_energy_original.png')
        save_mel_energy(get_mel_energy("processed.wav", sr=48000, n_fft=4096, hop_length=512),
                        sr=48000, output_filename='mel_energy_processed.png')

        return redirect(url_for('report', result=f'Upsampling complete! Processing time {processing_time:.2f} s, '
                                                 f'duration {duration:.2f} s'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/report/<result>')
def report(result):
    return render_template("report.html", result=result, cache_bust=int(time.time()))

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)