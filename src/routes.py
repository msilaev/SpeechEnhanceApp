from . import app
#from .result import Result
from flask import (render_template, request, url_for,
                   jsonify, redirect, send_from_directory)
#from src.features import Features
from src.denoise import Denoise
from src.upsample48 import Upsample48
import os
import joblib
from werkzeug.utils import secure_filename

import librosa as lb
import soundfile as sf

@app.route('/', methods=['GET', 'POST'])
def file_upload():

    if request.method == 'POST':

        file = request.files.get('file') or None

        if file.filename == '' or (not (file.filename.endswith('.wav') or file.filename.endswith('.flac'))):
            return jsonify({'error': 'Please upload a WAV or FLAC file'})

        max_file_size = 10 * 1024 * 1024
        # 10 MB (adjust this as needed)

        try:
            filename = secure_filename(file.filename)

            if filename != '':
            
                file_ext = os.path.splitext(filename)[1]

                file.save(os.path.join(app.root_path,
                                       app.config['UPLOAD_FOLDER'],
                                       "user.wav"))
                file.close()

                audio = (
                    os.path.join(os.path.join(app.root_path,
                                                  app.config['UPLOAD_FOLDER'],
                                              "user.wav")))

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


def process_audio(filename, model_file, output_filename, target_sr):
    try:
        audio_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
        y, sr = lb.load(audio_path, sr=app.config['SAMPLING_RATE'])
        os.remove(audio_path)  # Delete the original file

        model_path = os.path.join(app.root_path, app.config['MODEL_FOLDER'], model_file)
        # Assuming `Denoise` and `Upsample48` classes have a `.predict` method
        if model_file == 'denoise.onnx':
            processor = Denoise(y)
        elif model_file == 'upsample48.onnx':
            processor = Upsample48(y)
        else:
            raise ValueError("Invalid model specified.")

        #processed_audio, processing_time, duration = processor.predict(model_path)

        try:
            processed_audio, processing_time, duration = processor.predict(model_path)
        except Exception as a:
            raise RuntimeError(f"processor.predict failed: {str(a)}")

        output_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], output_filename)
        sf.write(output_path, processed_audio, target_sr)
        return output_path, processing_time, duration
    
    except Exception as e:
        raise RuntimeError(f"Audio processing failed: {str(e)}")


@app.route('/denoise')
def denoise(filename= "user.wav"):
    
    try:
        processed_file, processing_time, duration = process_audio(filename, 'denoise.onnx', 'processed.wav', app.config['SAMPLING_RATE'])

        # Redirect to the report page with details
        return redirect(url_for(
            'report', 
            result=f'Denoising complete! Processing time: {processing_time:.2f} s, Audio duration: {duration:.2f} s'
        ))
   
   
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upsample48')
def upsample48(filename = "user.wav"):
    try:
        processed_file, processing_time, duration = process_audio(filename, 'upsample48.onnx', 'processed.wav', 48000)
        return redirect(url_for('report', 
                                result=f'Upsampling complete! Processing time {processing_time:.2f} s, duration {duration:.2f} s'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/report/<result>')
def report(result):
    return render_template("report.html", result=result)


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)