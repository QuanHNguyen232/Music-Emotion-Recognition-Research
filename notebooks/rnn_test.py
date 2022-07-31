# %%

import tensorflow as tf
from tensorflow.python.keras import layers as L
from tensorflow.python.keras.models import Model



import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sounddevice as sd
import pandas as pd

from IPython import display

from mer.utils import get_spectrogram, \
  plot_spectrogram, \
  load_metadata, \
  plot_and_play, \
  preprocess_waveforms, \
  split_train_test

from mer.const import *

from mer.model import get_rnn_model

# %%

model = get_rnn_model()
model.summary()

# %%

_in = tf.random.normal((16, 128, 10000))
_label = tf.round(tf.random.normal((16, 1), mean=0.5, stddev=0.5))
_out = model(_in, training=False)


# %%


DATA_SEPARATION_PATH = "../data/PMEmo/PMEmo2019/PMEmo2019/separation"
DATA_SOURCE_PATH = "../data/PMEmo/PMEmo2019/PMEmo2019/chorus_wav"

# %%
datagen = os.listdir(DATA_SOURCE_PATH)



# %%

def train_datagen_song_level():
  """ Predicting valence mean and arousal mean
  """
  pointer = 0
  datagen = os.listdir(DATA_SOURCE_PATH)
  while True:
    # Reset pointer
    if pointer >= len(train_df):
      pointer = 0

    row = train_df.loc[pointer]
    song_id = row["song_id"]
    valence_mean = float(row["valence_mean"])
    arousal_mean = float(row["arousal_mean"])
    label = tf.convert_to_tensor([valence_mean, arousal_mean], dtype=tf.float32)
    song_path = os.path.join(AUDIO_FOLDER, str(int(song_id)) + SOUND_EXTENSION)
    audio_file = tf.io.read_file(song_path)
    waveforms, _ = tf.audio.decode_wav(contents=audio_file)
    waveforms = preprocess_waveforms(waveforms, WAVE_ARRAY_LENGTH)
    # print(waveforms.shape)

    # Work on building spectrogram
    # Shape (timestep, frequency, n_channel)
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
    pointer += 1

    padded_spectrogram = np.zeros((SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL), dtype=float)
    # spectrograms = spectrograms[tf.newaxis, ...]
    # some spectrogram are not the same shape
    padded_spectrogram[:spectrograms.shape[0], :spectrograms.shape[1], :] = spectrograms
    
    yield (tf.convert_to_tensor(padded_spectrogram), label)
