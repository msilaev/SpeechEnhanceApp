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

PATCH_SIZE = 8192

class Upsample16:

    def __init__(self, audio_data):

        self.y = audio_data
        #self.y = librosa.resample(audio_data, orig_sr = 16000, target_sr = 4000)

    def predict(self, model_path):

        x_noisy = self.y

        padding_needed = PATCH_SIZE - (x_noisy.shape[0] % PATCH_SIZE)

        x_noisy = np.pad(x_noisy, (0, padding_needed), 'constant', constant_values=(0, 0))

        x_noisy_spline = decimate(x_noisy, 4)
        x_noisy_spline = self.spline_up(x_noisy_spline, 4)

        n_patches = x_noisy_spline.shape[0] // PATCH_SIZE

        ort_session = ort.InferenceSession(model_path)

        P = []
        X = []

        start_time = time.time()

        for i in range(0, n_patches, 1):
            lr_patch = np.array(x_noisy_spline[i * PATCH_SIZE : (i+1)* PATCH_SIZE ])                  
         
            input_data = np.expand_dims(lr_patch, axis=0)  # Add batch dimension
            input_data = np.expand_dims(input_data, axis=2)  # Add channel dimension
                      
            inputs = {ort_session.get_inputs()[0].name: input_data}
                  
            output_name = ort_session.get_outputs()[0].name
            predictions = ort_session.run([output_name], inputs)

            P.append( np.squeeze(predictions))
            X.append(lr_patch)

            #print(np.squeeze(predictions).shape)            

        predictions = (np.array(np.concatenate(P))).flatten()
        input = (np.array(np.concatenate(X))).flatten()
        
        end_time = time.time()

        #print("duration",  librosa.get_duration(y=x_noisy, sr=16000))

        return predictions, x_noisy, end_time - start_time,  librosa.get_duration(y=predictions, sr=16000)
    
    @staticmethod
    def spline_up(x, r):
         x = x.flatten()
         len_x_up = len(x)*r
         x_up = np.zeros(len_x_up)

         i_lr = np.arange(len_x_up, step=r)
         i_hr = np.arange(len_x_up)

         f = interpolate.splrep(i_lr, x)

         x_sp = interpolate.splev(i_hr, f)

         return x_sp.astype(np.float32)

if __name__ == "__main__":

    audio = "../street_10dB/10dB/sp01_street_sn10.wav"
 
    y, sr = lb.load(audio, sr = 16000)

    x_noisy = y.flatten()

    model_path = "models/upsample16_gan_500.onnx"
    #model_path = "models/upsample16_gan_500.onnx"

    ort_session = ort.InferenceSession(model_path)

    n_patches = x_noisy.shape[0] // PATCH_SIZE

    x_noisy = np.pad(x_noisy, ((0, (n_patches+1)*PATCH_SIZE)), mode='constant')

    P = []
       
    for i in range(0, n_patches+1, 1):

        lr_patch = np.array(x_noisy[i * PATCH_SIZE : (i+1)* PATCH_SIZE ])
         
        input_data = np.expand_dims(lr_patch, axis=0)  # Add batch dimension
        input_data = np.expand_dims(input_data, axis=2)  # Add channel dimension
                      
        inputs = {ort_session.get_inputs()[0].name: input_data}
           
        output_name = ort_session.get_outputs()[0].name
        predictions = ort_session.run([output_name], inputs)

        P.append( np.squeeze(predictions))

        print(np.squeeze(predictions).shape)

    predictions = np.array(np.concatenate(P))
    
    sf.write("denoised.wav", predictions.flatten(), sr)
    sf.write("noisy.wav", y, sr)
        # The first outpu

    