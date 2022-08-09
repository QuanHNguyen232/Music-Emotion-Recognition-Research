#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 14, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Example notebook file
"""

# %%

import argparse
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
from tensorflow.python.keras import layers as L
from tensorflow.python.keras.models import Model

from mer.utils.const import get_config_from_json, setup_global_config

# Argument parsing
config_path = "../configs/config.json"
config = get_config_from_json(config_path)

##### Workaround to setup global config ############
setup_global_config(config, verbose=True)
from mer.utils.const import GLOBAL_CONFIG
##### End of Workaround #####

# Because the generator and some classes are based on the
# GLOBAL_CONFIG, we have to import them after we set the config
from mer.utils.utils import load_metadata, split_train_test, \
  preprocess_waveforms, get_spectrogram, plot_and_play, \
  pad_waveforms, plot_wave
from mer.model import get_rnn_model,get_rnn_model_2, Simple_CRNN_3
from mer.optimizer import get_Adam_optimizer, get_SGD_optimizer
from mer.loss import simple_mae_loss, simple_mse_loss


# %%%


filenames = tf.io.gfile.glob(str(GLOBAL_CONFIG.AUDIO_FOLDER) + '/*')
# Process with average annotation per song. 

df = load_metadata(GLOBAL_CONFIG.ANNOTATION_SONG_LEVEL)

# Smaller set of data
df = df[:64]

df

# %%

train_df, test_df = split_train_test(df, GLOBAL_CONFIG.TRAIN_RATIO)

# %%

# song_path = "../data/PMEmo/PMEmo2019/PMEmo2019/chorus_wav/6.wav"

# lib_wave, sr = librosa.load(song_path, GLOBAL_CONFIG.DEFAULT_FREQ)
# lib_wave = tf.convert_to_tensor(lib_wave)[..., tf.newaxis]

# plot_wave(lib_wave, second_length=40)


# audio_file = tf.io.read_file(song_path)

# waveforms, sample_rate = tf.audio.decode_wav(contents=audio_file)


# def plot_wave_4(waveforms, second_id = 0, second_length = 10, channel = 0):
#   from_id = int(44100 * second_id)
#   to_id = min(int(44100 * (second_id + second_length)), waveforms.shape[0])

#   fig, axes = plt.subplots(1, figsize=(12, 4))
#   timescale = np.arange(to_id - from_id)
#   axes.plot(timescale, waveforms[from_id:to_id, channel].numpy())
#   axes.set_title('Waveform')
#   axes.set_xlim([0, int(44100 * second_length)])
#   plt.show()

# plot_wave_4(waveforms, second_length=40)



# %%

def load_wave_data(song_path):
  # audio_file = tf.io.read_file(song_path)
  # waveforms, sample_rate = tf.audio.decode_wav(contents=audio_file)
  # waveforms = tfio.audio.resample(waveforms, sample_rate, GLOBAL_CONFIG.DEFAULT_FREQ)
  waveforms, sample_rate = librosa.load(song_path, GLOBAL_CONFIG.DEFAULT_FREQ)
  waveforms = tf.convert_to_tensor(waveforms)[..., tf.newaxis]
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

def train_datagen():
  """ Predicting valence mean and arousal mean
  """
  pointer = 0
  while True:
    # Reset pointer
    if pointer >= len(train_df):
      pointer = 0

    row = train_df.loc[pointer]
    song_id = row["musicId"]
    arousal_mean = float(row["Arousal(mean)"])
    valence_mean = float(row["Valence(mean)"])
    
    # TODO: HOw are we gonna integrate valence and arousal std?
    arousal_std = float(row["Arousal(std)"])
    valence_std = float(row["Valence(std)"])
    
    label = tf.convert_to_tensor([valence_mean, arousal_mean, valence_std, arousal_std], dtype=tf.float32)
    song_path = os.path.join(GLOBAL_CONFIG.AUDIO_FOLDER, str(int(song_id)) + GLOBAL_CONFIG.SOUND_EXTENSION)
    
    waveforms = load_wave_data(song_path)
    
    spectrogram_features = extract_spectrogram_features(waveforms)
    
    # Preprocessed and normalize waveforms in the end
    waveforms = preprocess_waveforms(waveforms)

    # Update pointer
    pointer += 1
    yield (waveforms, spectrogram_features, label)

def test_datagen():
  """ Predicting valence mean and arousal mean
  """
  pointer = 0
  while True:
    # Reset pointer
    if pointer >= len(test_df):
      pointer = 0

    row = test_df.loc[pointer]
    song_id = row["musicId"]
    valence_mean = float(row["Valence(mean)"])
    arousal_mean = float(row["Arousal(mean)"])
    arousal_std = float(row["Arousal(std)"])
    valence_std = float(row["Valence(std)"])
    label = tf.convert_to_tensor([valence_mean, arousal_mean, valence_std, arousal_std], dtype=tf.float32)
    song_path = os.path.join(GLOBAL_CONFIG.AUDIO_FOLDER, str(int(song_id)) + GLOBAL_CONFIG.SOUND_EXTENSION)

    waveforms = load_wave_data(song_path)
    
    spectrogram_features = extract_spectrogram_features(waveforms)
    
    # Preprocessed and normalize waveforms in the end
    waveforms = preprocess_waveforms(waveforms)

    pointer += 1
    yield (waveforms, spectrogram_features, label)

train_dataset = tf.data.Dataset.from_generator(
  train_datagen,
  output_signature=(
    tf.TensorSpec(shape=(GLOBAL_CONFIG.WAVE_ARRAY_LENGTH, GLOBAL_CONFIG.N_CHANNEL), dtype=tf.float32),
    tf.TensorSpec(shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL), dtype=tf.float32),
    tf.TensorSpec(shape=(4), dtype=tf.float32)
  )
)
train_batch_dataset = train_dataset.batch(GLOBAL_CONFIG.BATCH_SIZE)
# train_batch_dataset = train_batch_dataset.cache().prefetch(tf.data.AUTOTUNE) # OOM error
train_batch_iter = iter(train_batch_dataset)


# Comment out to decide to create a normalization layer.
# NOTE: this is every time consuming because it looks at all the data, only 
# use this at the first time.
# NOTE: Normally, we create this layer once, save it somewhere to reuse in
# every other model.
#
# norm_layer = L.Normalization()
# norm_layer.adapt(data=train_dataset.map(map_func=lambda spec, label: spec))
#

test_dataset = tf.data.Dataset.from_generator(
  test_datagen,
  output_signature=(
    tf.TensorSpec(shape=(GLOBAL_CONFIG.WAVE_ARRAY_LENGTH, GLOBAL_CONFIG.N_CHANNEL), dtype=tf.float32),
    tf.TensorSpec(shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL), dtype=tf.float32),
    tf.TensorSpec(shape=(4), dtype=tf.float32)
  )
)
test_batch_dataset = test_dataset.batch(GLOBAL_CONFIG.BATCH_SIZE)
# test_batch_dataset = test_batch_dataset.cache().prefetch(tf.data.AUTOTUNE) # OOM error
test_batch_iter = iter(test_batch_dataset)

# %%
# test data gen

in_wave, in_spec, out = next(test_batch_iter)
print(in_wave.shape)
print(in_spec.shape)
print(out.shape)

# plot_wave(in_wave[0], second_id = 0, second_length = 40, channel = 0)
# plot_and_play(in_wave[5], second_id = 0, second_length = 40, channel = 0)



# %%

# model = get_rnn_model()
# model.summary()
# model_name = "rnn_1"
# sample_input = tf.ones(shape=(BATCH_SIZE, SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2))
# with tf.device("/CPU:0"):
#   sample_output = model(sample_input, training=False)
# print(sample_output)

# model = get_rnn_model_2(input_shape=in_wave.shape[1:])
# model.summary()
# model_name = "rnn_2"

model = Simple_CRNN_3(input_shape=in_spec.shape[1:])
model.summary()
model_name = "crnn_3"

# %%


history_path = f"../history/{model_name}.npy"
weights_path = f"../models/{model_name}/checkpoint"

# optimizer = get_SGD_optimizer()
optimizer = get_Adam_optimizer()

# %%

from mer.trainer import Trainer

trainer = Trainer(model,
  train_batch_iter,
  test_batch_iter,
  optimizer,
  simple_mse_loss,
  epochs=10,
  steps_per_epoch=4, # 64 // 16
  valid_step=2,
  history_path=history_path,
  weights_path=weights_path,
  save_history=True)

# About 50 epochs with each epoch step 100 will cover the whole training dataset!
history = trainer.train()

# %%

from mer.utils.utils import plot_history

plot_history(history_path)

# %%

def evaluate(df_pointer, model, loss_func, play=False):
  row = test_df.loc[df_pointer]
  song_id = row["musicId"]
  arousal_mean = float(row["Arousal(mean)"])
  valence_mean = float(row["Valence(mean)"])
  arousal_std = float(row["Arousal(std)"])
  valence_std = float(row["Valence(std)"])
  label = tf.convert_to_tensor([valence_mean, arousal_mean, valence_std, arousal_std], dtype=tf.float32)
  print(f"Label: Valence: {valence_mean}, Arousal: {arousal_mean}")
  song_path = os.path.join(GLOBAL_CONFIG.AUDIO_FOLDER, str(int(song_id)) + GLOBAL_CONFIG.SOUND_EXTENSION)
  
  waveforms = load_wave_data(song_path)
  waveforms = pad_waveforms(waveforms, GLOBAL_CONFIG.WAVE_ARRAY_LENGTH)
  spectrograms = extract_spectrogram_features(waveforms)[tf.newaxis, ...]
  waveforms = preprocess_waveforms(waveforms)

  ## Eval
  y_pred = model(spectrograms, training=False)[0]

  print(y_pred.shape)

  print(f"Predicted y_pred value: Valence: {y_pred[0]}, Arousal: {y_pred[1]}")

  loss = loss_func(label[tf.newaxis, ...], y_pred)
  print(f"Loss: {loss}")

  if play:
    plot_and_play(waveforms, 0, 40, 0)

i = 0

# %%

# model.load_weights(weights_path)

# %%

i += 1
evaluate(i, model, simple_mse_loss, play=True)


