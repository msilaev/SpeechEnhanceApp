from . import app
from flask import (render_template, request, url_for,
                   jsonify, redirect, send_from_directory)

from src.denoise import Denoise
from src.upsample48 import Upsample48
from src.upsample16 import Upsample16

import os
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
    
    return output_path



def process_audio(filename, model_file, output_filename, target_sr):

    try:
        audio_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)

        y, sr = lb.load(audio_path, sr=app.config['SAMPLING_RATE'])
        
        model_path = os.path.join(app.root_path, app.config['MODEL_FOLDER'], model_file)

        if model_file == 'denoise.onnx':
            processor = Denoise(y)
        elif model_file == 'upsample48.onnx':
            processor = Upsample48(y)
        elif model_file == 'upsample16.onnx':
            processor = Upsample16(y)    
        else:
            raise ValueError("Invalid model specified.")

        try:
            processed_audio, input_audio, processing_time, duration = processor.predict(model_path)
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

        file = request.files.get('file') or None

        if file.filename == '' or (not (file.filename.endswith('.wav') or file.filename.endswith('.flac'))):
            return jsonify({'error': 'Please upload a WAV or FLAC file'})

        max_file_size = 10 * 1024 * 1024

        try:
            filename = secure_filename(file.filename)

            if filename != '':
            
                file_ext = os.path.splitext(filename)[1]

                file.save(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], "user.wav"))

                file.close()

                audio = os.path.join(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], "user.wav"))

                file_size = os.path.getsize(audio)

                if (file_size > max_file_size):
                    return jsonify({'error': "File size exceeds the limit of 10 MB!"})
            
            return redirect(url_for('upload_complete', result = "Upload complete" ))
           
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('upload.html')


@app.route('/upload_complete/<result>')
def upload_complete(result):   
   return render_template("upload_complete.html", result = result)


@app.route('/denoise')
def denoise(filename= "user.wav"):
    
    try:

        processed_file, processing_time, duration = process_audio(filename, 'denoise.onnx', 'processed.wav', app.config['SAMPLING_RATE'])

        original_file_image = save_spectrum( get_spectrum("user.wav", sr= 16000, n_fft=2048), sr = 16000, hop_length=2048 // 4,
            output_filename = 'spectrogram_original.png', type='original')          
        
        processed_file_image = save_spectrum( get_spectrum("processed.wav", sr= 16000, n_fft=2048), sr = 16000, hop_length=2048 // 4,
            output_filename = 'spectrogram_processed.png', type='original')  

        return redirect(url_for('report', result=f'Denoising complete! Processing time: {processing_time:.2f} s, Audio duration: {duration:.2f} s' ))
      
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upsample48')
def upsample48(filename = "user.wav"):
    
    try:
        
        processed_file, processing_time, duration = process_audio(filename, 'upsample48.onnx', 'processed.wav', 48000)
        
        original_file_image = save_spectrum( get_spectrum("user.wav", sr = 48000, n_fft=3*2048), sr = 48000, hop_length=3*2048 // 4,
            output_filename = 'spectrogram_original.png', type='original')          
        
        processed_file_image = save_spectrum( get_spectrum("processed.wav", sr = 48000, n_fft=3*2048), sr = 48000, hop_length=3*2048 // 4,
            output_filename = 'spectrogram_processed.png', type='original')  
        
        return redirect(url_for('report', 
                                result=f'Upsampling complete! Processing time {processing_time:.2f} s, duration {duration:.2f} s'))
       
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upsample16')
def upsample16(filename = "user.wav"):

    try:

        processed_file, processing_time, duration = process_audio(filename, 'upsample16.onnx', 'processed.wav', 16000)       

        original_file_image = save_spectrum( get_spectrum("user.wav", sr = 16000 , n_fft=2048), sr = 16000, hop_length=2048 // 4,
            output_filename = 'spectrogram_original.png', type='original')          
        
        processed_file_image = save_spectrum( get_spectrum("processed.wav", sr = 16000, n_fft=2048), sr = 16000, hop_length=2048 // 4,
            output_filename = 'spectrogram_processed.png', type='original')  
        
        return redirect(url_for('report', result=f'Upsampling complete! Processing time {processing_time:.2f} s, duration {duration:.2f} s'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/report/<result>')
def report(result):
    return render_template("report.html", result=result)


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)