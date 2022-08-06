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

import tensorflow as tf

print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import argparse
import os
import numpy as np

from mer.utils.const import get_config_from_json, setup_global_config

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
  preprocess_waveforms, get_spectrogram
from mer.model import get_rnn_model

from tensorflow.python.keras import layers as L
from tensorflow.python.keras.models import Model

# %%%


filenames = tf.io.gfile.glob(str(GLOBAL_CONFIG.AUDIO_FOLDER) + '/*')
# Process with average annotation per song. 

df = load_metadata(GLOBAL_CONFIG.ANNOTATION_SONG_LEVEL)

df

# %%

train_df, test_df = split_train_test(df, GLOBAL_CONFIG.TRAIN_RATIO)

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
    # valence_std = float(row["Arousal(std)"])
    # arousal_std = float(row["Valence(std)"])
    
    label = tf.convert_to_tensor([valence_mean, arousal_mean], dtype=tf.float32)
    song_path = os.path.join(GLOBAL_CONFIG.AUDIO_FOLDER, str(int(song_id)) + GLOBAL_CONFIG.SOUND_EXTENSION)
    audio_file = tf.io.read_file(song_path)
    waveforms, _ = tf.audio.decode_wav(contents=audio_file)
    waveforms = preprocess_waveforms(waveforms, GLOBAL_CONFIG.WAVE_ARRAY_LENGTH)
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

    padded_spectrogram = np.zeros((GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL), dtype=float)
    # spectrograms = spectrograms[tf.newaxis, ...]
    # some spectrogram are not the same shape
    padded_spectrogram[:spectrograms.shape[0], :spectrograms.shape[1], :] = spectrograms
    
    yield (tf.convert_to_tensor(padded_spectrogram), label)

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
    label = tf.convert_to_tensor([valence_mean, arousal_mean], dtype=tf.float32)
    song_path = os.path.join(GLOBAL_CONFIG.AUDIO_FOLDER, str(int(song_id)) + GLOBAL_CONFIG.SOUND_EXTENSION)
    audio_file = tf.io.read_file(song_path)
    waveforms, _ = tf.audio.decode_wav(contents=audio_file)
    waveforms = preprocess_waveforms(waveforms, GLOBAL_CONFIG.WAVE_ARRAY_LENGTH)
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

    padded_spectrogram = np.zeros((GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL), dtype=float)
    # spectrograms = spectrograms[tf.newaxis, ...]
    # some spectrogram are not the same shape
    padded_spectrogram[:spectrograms.shape[0], :spectrograms.shape[1], :] = spectrograms
    
    yield (tf.convert_to_tensor(padded_spectrogram), label)

train_dataset = tf.data.Dataset.from_generator(
  train_datagen,
  output_signature=(
    tf.TensorSpec(shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL), dtype=tf.float32),
    tf.TensorSpec(shape=(2), dtype=tf.float32)
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
    tf.TensorSpec(shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL), dtype=tf.float32),
    tf.TensorSpec(shape=(2, ), dtype=tf.float32)
  )
)
test_batch_dataset = test_dataset.batch(GLOBAL_CONFIG.BATCH_SIZE)
# test_batch_dataset = test_batch_dataset.cache().prefetch(tf.data.AUTOTUNE) # OOM error
test_batch_iter = iter(test_batch_dataset)


# %%

# test data gen

_in, _out = next(test_batch_iter)

# %%

_in.shape

# %%

_out.shape


# %%


model = get_rnn_model()
model.summary()
model_name = "rnn_1"
# sample_input = tf.ones(shape=(BATCH_SIZE, SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2))
# with tf.device("/CPU:0"):
#   sample_output = model(sample_input, training=False)
# print(sample_output)

# %%


history_path = f"../history/{model_name}.npy"
weights_path = f"../models/{model_name}/checkpoint"

from mer.optimizer import get_SGD_optimizer
from mer.loss import simple_mae_loss, simple_mse_loss
optimizer = get_SGD_optimizer()

# %%

from mer.trainer import Trainer

trainer = Trainer(model,
  train_batch_iter,
  test_batch_iter,
  optimizer,
  simple_mae_loss,
  epochs=2,
  steps_per_epoch=100, # 1800 // 16
  valid_step=20,
  history_path=history_path,
  weights_path=weights_path,
  save_history=True)

# About 50 epochs with each epoch step 100 will cover the whole training dataset!
history = trainer.train()

# %%
