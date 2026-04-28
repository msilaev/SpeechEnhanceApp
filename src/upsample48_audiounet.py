import librosa as lb
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy import interpolate
from scipy.signal import decimate
import librosa
import time

PATCH_SIZE = 8192

class Upsample48AudioUnet:

    def __init__(self, audio_data):
        self.y = audio_data

    def predict(self, ort_session, skip_decimate=False):

        x_noisy = self.y

        padding_needed = PATCH_SIZE - (x_noisy.shape[0] % PATCH_SIZE)
        x_noisy = np.pad(x_noisy, (0, padding_needed), 'constant', constant_values=(0, 0))

        if skip_decimate:
            x_noisy_spline = self.spline_up(x_noisy, 3)
        else:
            x_noisy_spline = decimate(x_noisy, 3)
            x_noisy_spline = self.spline_up(x_noisy_spline, 3)

        n_patches = x_noisy_spline.shape[0] // PATCH_SIZE

        P = []
        X = []

        start_time = time.time()

        for i in range(n_patches):
            lr_patch = np.array(x_noisy_spline[i * PATCH_SIZE : (i + 1) * PATCH_SIZE])

            input_data = np.expand_dims(lr_patch, axis=0)
            input_data = np.expand_dims(input_data, axis=2)

            inputs = {ort_session.get_inputs()[0].name: input_data}
            output_name = ort_session.get_outputs()[0].name
            predictions = ort_session.run([output_name], inputs)

            P.append(np.squeeze(predictions))
            X.append(lr_patch)

        predictions = (np.array(np.concatenate(P))).flatten()

        end_time = time.time()

        return predictions, x_noisy_spline, end_time - start_time, librosa.get_duration(y=predictions, sr=48000)

    @staticmethod
    def spline_up(x, r):
        x = x.flatten()
        len_x_up = len(x) * r
        i_lr = np.arange(len_x_up, step=r)
        i_hr = np.arange(len_x_up)
        f = interpolate.splrep(i_lr, x)
        return interpolate.splev(i_hr, f).astype(np.float32)
