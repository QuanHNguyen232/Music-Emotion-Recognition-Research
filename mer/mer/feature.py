#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 13, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Example dataloader
"""

import tensorflow as tf
import os
import librosa
import numpy as np

from .utils.utils import pad_waveforms, get_spectrogram

from .utils.const import GLOBAL_CONFIG

# def load_wave_data(song_path):
#   # audio_file = tf.io.read_file(song_path)
#   # waveforms, sample_rate = tf.audio.decode_wav(contents=audio_file)
#   # waveforms = tfio.audio.resample(waveforms, sample_rate, GLOBAL_CONFIG.DEFAULT_FREQ)
#   # waveforms, sample_rate = librosa.load(song_path, GLOBAL_CONFIG.DEFAULT_FREQ)
#   waveforms, sample_rate = librosa.load(song_path)
#   waveforms = tf.convert_to_tensor(waveforms)[..., tf.newaxis]
#   waveforms = pad_waveforms(waveforms, GLOBAL_CONFIG.WAVE_ARRAY_LENGTH)
#   return waveforms

def load_wave_data(song_path):
  audio_file = tf.io.read_file(song_path)
  waveforms, _ = tf.audio.decode_wav(contents=audio_file, desired_channels=1)
  waveforms = pad_waveforms(waveforms, GLOBAL_CONFIG.WAVE_ARRAY_LENGTH)
  return waveforms

def extract_spectrogram_features(waveforms):
  spectrograms = None
  # Loop through each channel
  for i in range(waveforms.shape[-1]):
    # Shape (timestep, frequency, 1)
    spectrogram = get_spectrogram(waveforms[..., i], input_len=waveforms.shape[0])
    # spectrogram = tf.convert_to_tensor(np.log(spectrogram.numpy() + np.finfo(float).eps))
    if spectrograms == None:
      spectrograms = spectrogram
    else:
      spectrograms = tf.concat([spectrograms, spectrogram], axis=-1)
  
  padded_spectrogram = np.zeros((GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL), dtype=float)

  # some spectrogram are not the same shape
  padded_spectrogram[:spectrograms.shape[0], :spectrograms.shape[1], :] = spectrograms
  return tf.convert_to_tensor(padded_spectrogram)
