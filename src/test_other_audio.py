import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from models.gan import Generator
from models.audiounet import AudioUNet
from models.io import inference_wav_other_audio

from models.io import inference_wav
import soundfile as sf

import argparse

# -----------------------------------------------------------------
def make_parser():

  parser = argparse.ArgumentParser()
    # train
  parser.add_argument('--model')
  parser.add_argument('--logname')
  parser.add_argument('--output')
  parser.add_argument('--data')
  parser.add_argument('--sr', type=int)
  parser.add_argument('--patch_size', type=int, default=8192,
                            help='Size of patches over which the model operates')

  return parser

def eval(args):

  if args.model in [ "gan_multispeaker"]:

    #model = GeneratorDilConv(layers=5)
    model = Generator(layers=5)

  elif args.model in ["audiounet_multispeaker"]:

    model = AudioUNet(layers=4)

  model.eval()

  checkpoint_root = args.logname
  output_dir = args.output
  directory = args.data

  for root, dirs, files in os.walk(directory):
      for file in files:

          if file.endswith(".flac") or file.endswith(".wav") or file.endswith(".mp3"):

              input_file_path = os.path.join(root, file)

              input_file_path_wav = os.path.join(root, file.split(".")[0])

              relative_path_wav = os.path.relpath(input_file_path_wav, directory)

              output_file_path = os.path.join(output_dir, relative_path_wav + ".pr.wav")

              output_dir_1 = os.path.dirname(output_file_path)
              os.makedirs(output_dir_1, exist_ok = True)

              print(input_file_path)
              print(output_file_path)

              P, X = inference_wav_other_audio(model, input_file_path,
                                      args, epoch=None, model_path=checkpoint_root)

              x_pr = P.flatten()

              sf.write(output_file_path + '.wav', x_pr, args.sr)

def main():
    parser = make_parser()
    args = parser.parse_args()
    eval(args)


if __name__ == "__main__":
    main()