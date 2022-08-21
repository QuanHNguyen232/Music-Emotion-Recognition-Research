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
from mer.feature import load_wave_data, extract_spectrogram_features


# %%%


filenames = tf.io.gfile.glob(str(GLOBAL_CONFIG.AUDIO_FOLDER) + '/*')
# Process with average annotation per song. 

df = load_metadata(GLOBAL_CONFIG.ANNOTATION_SONG_LEVEL)

# Smaller set of data
# df = df[:64]

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


#%%
def train_sep_datagen():
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
    song_dir = os.path.join(GLOBAL_CONFIG.SEP_AUDIO_FOLDER, str(int(song_id)) + GLOBAL_CONFIG.MP3_EXTENSION)

    waveform_list = []
    spectrogram_list = []
    for wav_file in os.listdir(song_dir):
      song_path = os.path.join(song_dir, wav_file)
      
      sep_waveform = load_wave_data(song_path)
      sep_spectrogram_feat = extract_spectrogram_features(sep_waveform)
      sep_waveform = preprocess_waveforms(sep_waveform)

      waveform_list.append(sep_waveform)
      spectrogram_list.append(sep_spectrogram_feat)
    
    
    waveforms = tf.concat(waveform_list, axis=1)  # shape=(x, 1) --> (x, 4)
    spectrograms = tf.concat(spectrogram_list, axis=2)  # shape=(h, w, 1) --> (h, w, 4)

    # Update pointer
    pointer += 1
    yield (waveforms, spectrograms, label)

def test_sep_datagen():
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

    song_dir = os.path.join(GLOBAL_CONFIG.SEP_AUDIO_FOLDER, str(int(song_id)) + GLOBAL_CONFIG.MP3_EXTENSION)
    waveform_list = []
    spectrogram_list = []
    for wav_file in os.listdir(song_dir):
      song_path = os.path.join(song_dir, wav_file)
      
      sep_waveform = load_wave_data(song_path)
      sep_spectrogram_feat = extract_spectrogram_features(sep_waveform)
      sep_waveform = preprocess_waveforms(sep_waveform)

      waveform_list.append(sep_waveform)
      spectrogram_list.append(sep_spectrogram_feat)
    
    waveforms = tf.concat(waveform_list, axis=1)  # shape=(x, 1) --> (x, 4)
    spectrograms = tf.concat(spectrogram_list, axis=2)  # shape=(h, w, 1) --> (h, w, 4)

    pointer += 1
    yield (waveforms, spectrograms, label)

# Create train dataset on seperated 16bit
train_sep_dataset = tf.data.Dataset.from_generator(
  train_sep_datagen,
  output_signature=(
    tf.TensorSpec(shape=(GLOBAL_CONFIG.WAVE_ARRAY_LENGTH, GLOBAL_CONFIG.N_CHANNEL_SEP), dtype=tf.float32),
    tf.TensorSpec(shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL_SEP), dtype=tf.float32),
    tf.TensorSpec(shape=(4), dtype=tf.float32)
  )
)
train_batch_sep_dataset = train_sep_dataset.batch(GLOBAL_CONFIG.BATCH_SIZE)
train_batch_iter = iter(train_batch_sep_dataset)

# Create test dataset on seperated 16bit
test_sep_dataset = tf.data.Dataset.from_generator(
  test_sep_datagen,
  output_signature=(
    tf.TensorSpec(shape=(GLOBAL_CONFIG.WAVE_ARRAY_LENGTH, GLOBAL_CONFIG.N_CHANNEL_SEP), dtype=tf.float32),
    tf.TensorSpec(shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL_SEP), dtype=tf.float32),
    tf.TensorSpec(shape=(4), dtype=tf.float32)
  )
)
test_batch_sep_dataset = test_sep_dataset.batch(GLOBAL_CONFIG.BATCH_SIZE)
test_batch_sep_iter = iter(test_batch_sep_dataset)

# %%%%
# test data gen

in_wave, in_spec, out = next(test_batch_sep_iter)
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


history_path = f"../history/{model_name}_sep.npy"
weights_path = f"../models/{model_name}_sep/checkpoint"

# optimizer = get_SGD_optimizer()
optimizer = get_Adam_optimizer()

# %%

from mer.trainer import Trainer

trainer = Trainer(model,
  train_batch_iter,
  test_batch_sep_iter,
  optimizer,
  simple_mse_loss,
  epochs=5,
  steps_per_epoch=90, # // 64 // 16 // //////     724 // 16 = 45
  valid_step=30,
  history_path=history_path,
  weights_path=weights_path,
  save_history=True)

# About 50 epochs with each epoch step 100 will cover the whole training dataset!
history = trainer.train()

# %%

from mer.utils.utils import plot_history

plot_history(history_path)

other_path = f"../history/{model_name}.npy"
plot_history(other_path)


# %%

import matplotlib.pyplot as plt
# Plot
with open(history_path, "rb") as f:
  [sep_epochs_loss, sep_epochs_val_loss] = np.load(f, allow_pickle=True)

with open(other_path, "rb") as f:
  [epochs_loss, epochs_val_loss] = np.load(f, allow_pickle=True)


# %%


print("Normal source val loss:", epochs_val_loss[3])
print("Separated source val loss:", sep_epochs_val_loss[4])












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
  print(song_path)
  waveforms = load_wave_data(song_path)
  spectrograms = extract_spectrogram_features(waveforms)[tf.newaxis, ...]
  # waveforms = preprocess_waveforms(waveforms)

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
evaluate(i, model, simple_mse_loss, play=False)



# %%
