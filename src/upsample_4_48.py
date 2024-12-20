import librosa as lb
import numpy as np
from pathlib import Path
import joblib
import onnx
import onnxruntime as ort
import soundfile as sf
from scipy import interpolate
from scipy.signal import decimate
import librosa
import time

from .upsample16 import Upsample16
from .upsample48 import Upsample48

PATCH_SIZE = 8192

class Upsample448:

    def __init__(self, audio_data):
        self.y = audio_data

    def predict(self, model_path_16, model_path_48):
        x_noisy = self.y
        x_noisy_spline = decimate(x_noisy, 4)
        x_noisy = librosa.resample(x_noisy_spline, orig_sr=4000, target_sr=48000)
        #x_noisy = librosa.resample(x_noisy, orig_sr=16000, target_sr=48000)

        start_time = time.time()

        processor = Upsample16(self.y)
        processed_audio, input_audio, processing_time, duration = processor.predict(model_path_16)

        processor = Upsample48(processed_audio)
        processed_audio, input_audio, processing_time, duration = processor.predict(model_path_48)

        end_time = time.time()

        return processed_audio, x_noisy, end_time - start_time, librosa.get_duration(y=self.y, sr=16000)

#       return predictions, x_noisy, end_time - start_time,  librosa.get_duration(y=predictions, sr=48000)

if __name__ == "__main__":

    audio = "../street_10dB/10dB/sp01_street_sn10.wav"

    y, sr = lb.load(audio, sr=48000)

    x_noisy = y.flatten()

    model_path_16 = "models/upsample16_gan_500.onnx"
    model_path_48 = "models/upsample48_gan_500.onnx"

    ort_session = ort.InferenceSession(model_path_16)

    n_patches = x_noisy.shape[0] // PATCH_SIZE

    x_noisy = np.pad(x_noisy, ((0, (n_patches + 1) * PATCH_SIZE)), mode='constant')

    P = []

    for i in range(0, n_patches + 1, 1):
        lr_patch = np.array(x_noisy[i * PATCH_SIZE: (i + 1) * PATCH_SIZE])

        input_data = np.expand_dims(lr_patch, axis=0)  # Add batch dimension
        input_data = np.expand_dims(input_data, axis=2)  # Add channel dimension

        inputs = {ort_session.get_inputs()[0].name: input_data}

        output_name = ort_session.get_outputs()[0].name
        predictions = ort_session.run([output_name], inputs)

        P.append(np.squeeze(predictions))

        print(np.squeeze(predictions).shape)

    predictions = np.array(np.concatenate(P))

    sf.write("denoised.wav", predictions.flatten(), sr)
    sf.write("noisy.wav", y, sr)
    # The first outpu
