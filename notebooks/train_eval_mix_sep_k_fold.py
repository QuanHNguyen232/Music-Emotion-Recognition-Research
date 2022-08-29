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
from pickle import FALSE
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd

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
from mer.utils.utils import compute_all_kl_divergence, load_metadata, split_train_test, \
  preprocess_waveforms, get_spectrogram, plot_and_play, \
  pad_waveforms, plot_wave
from mer.model import get_rnn_model,get_rnn_model_2, Simple_CRNN_3
from mer.optimizer import get_Adam_optimizer, get_SGD_optimizer
from mer.loss import simple_mae_loss, simple_mse_loss
from mer.feature import load_wave_data, extract_spectrogram_features


columns = [
  "song_id", "gt_valence_mean", "gt_arousal_mean",
  "gt_valence_std", "gt_arousal_std", "mixed_valence_mean", 
  "mixed_arousal_mean", "mixed_valence_std", "mixed_arousal_std",
  "sep_valence_mean", "sep_arousal_mean", "sep_valence_std",
  "sep_arousal_std"
]
os.makedirs("./results", exist_ok=True)
os.makedirs("./kl_results", exist_ok=True)
filenames = tf.io.gfile.glob(str(GLOBAL_CONFIG.AUDIO_FOLDER) + '/*')
# Process with average annotation per song. 

# df = load_metadata(GLOBAL_CONFIG.ANNOTATION_SONG_LEVEL) # This is to train on only one fold

# K-fold training
for fold, fname in enumerate(os.listdir(GLOBAL_CONFIG.K_FOLD_ANNOTATION_FOLDER)):
  print(f"---------- Procerss training and evaluating for fold {fold} -------------")
  df = pd.read_csv(os.path.join(GLOBAL_CONFIG.K_FOLD_ANNOTATION_FOLDER, fname))

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
    while pointer < len(test_df):

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
      yield (song_id, waveforms, spectrogram_features, label)

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
    while pointer < len(test_df):

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
      yield (song_id, waveforms, spectrograms, label)

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

  test_dataset = tf.data.Dataset.from_generator(
    test_datagen,
    output_signature=(
      tf.TensorSpec(shape=(), dtype=tf.int32),
      tf.TensorSpec(shape=(GLOBAL_CONFIG.WAVE_ARRAY_LENGTH, GLOBAL_CONFIG.N_CHANNEL), dtype=tf.float32),
      tf.TensorSpec(shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL), dtype=tf.float32),
      tf.TensorSpec(shape=(4), dtype=tf.float32)
    )
  )
  test_batch_dataset = test_dataset.batch(GLOBAL_CONFIG.BATCH_SIZE)
  # test_batch_dataset = test_batch_dataset.cache().prefetch(tf.data.AUTOTUNE) # OOM error
  test_batch_iter = iter(test_batch_dataset)

  def val_datagen():
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

  val_dataset = tf.data.Dataset.from_generator(
    val_datagen,
    output_signature=(
      tf.TensorSpec(shape=(GLOBAL_CONFIG.WAVE_ARRAY_LENGTH, GLOBAL_CONFIG.N_CHANNEL), dtype=tf.float32),
      tf.TensorSpec(shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL), dtype=tf.float32),
      tf.TensorSpec(shape=(4), dtype=tf.float32)
    )
  )
  val_batch_dataset = val_dataset.batch(GLOBAL_CONFIG.BATCH_SIZE)
  # test_batch_dataset = test_batch_dataset.cache().prefetch(tf.data.AUTOTUNE) # OOM error
  val_batch_iter = iter(val_batch_dataset)

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
  train_batch_sep_iter = iter(train_batch_sep_dataset)

  # Create test dataset on seperated 16bit
  test_sep_dataset = tf.data.Dataset.from_generator(
    test_sep_datagen,
    output_signature=(
      tf.TensorSpec(shape=(), dtype=tf.int32),
      tf.TensorSpec(shape=(GLOBAL_CONFIG.WAVE_ARRAY_LENGTH, GLOBAL_CONFIG.N_CHANNEL_SEP), dtype=tf.float32),
      tf.TensorSpec(shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL_SEP), dtype=tf.float32),
      tf.TensorSpec(shape=(4), dtype=tf.float32)
    )
  )
  test_batch_sep_dataset = test_sep_dataset.batch(GLOBAL_CONFIG.BATCH_SIZE)
  test_batch_sep_iter = iter(test_batch_sep_dataset)

  def val_sep_datagen():
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

  val_sep_dataset = tf.data.Dataset.from_generator(
    val_sep_datagen,
    output_signature=(
      tf.TensorSpec(shape=(GLOBAL_CONFIG.WAVE_ARRAY_LENGTH, GLOBAL_CONFIG.N_CHANNEL_SEP), dtype=tf.float32),
      tf.TensorSpec(shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL_SEP), dtype=tf.float32),
      tf.TensorSpec(shape=(4), dtype=tf.float32)
    )
  )
  val_batch_sep_dataset = val_sep_dataset.batch(GLOBAL_CONFIG.BATCH_SIZE)
  val_batch_sep_iter = iter(val_batch_sep_dataset)

  model_name = f"crnn_3_fold_{fold}"

  # Model mixed
  model_mixed = Simple_CRNN_3(input_shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL))
  history_mixed_path = f"../history/{model_name}.npy"
  weights_mixed_path = f"../models/{model_name}/checkpoint"
  # model_mixed.load_weights(weights_mixed_path)

  # Model sep
  model_sep = Simple_CRNN_3(input_shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL_SEP))
  history_sep_path = f"../history/{model_name}_sep.npy"
  weights_sep_path = f"../models/{model_name}_sep/checkpoint"
  # model_sep.load_weights(weights_sep_path)

  optimizer = get_Adam_optimizer()

  from mer.trainer import Trainer

  print("Training model mixed ...")
  trainer_mixed = Trainer(model_mixed,
    train_batch_iter,
    val_batch_iter,
    optimizer,
    simple_mse_loss,
    epochs=4,
    steps_per_epoch=77, #      613 // 8 = 77
    valid_step=7,
    history_path=history_mixed_path,
    weights_path=weights_mixed_path,
    save_history=True)

  # # About 50 epochs with each epoch step 100 will cover the whole training dataset!
  history = trainer_mixed.train()

  print("Training model sep ...")
  trainer_sep = Trainer(model_sep,
    train_batch_sep_iter,
    val_batch_sep_iter,
    optimizer,
    simple_mse_loss,
    epochs=4,
    steps_per_epoch=77, #      613 // 8 = 77
    valid_step=7,
    history_path=history_sep_path,
    weights_path=weights_sep_path,
    save_history=True)

  # About 50 epochs with each epoch step 100 will cover the whole training dataset!
  history2 = trainer_sep.train()

  csv_path_result = f"./results/result_fold_{fold}.csv"

  # method that use a model to infer and return prediction data
  def infer(test_data_iter, model):
    print(f"Inferring current fold {fold}")

    batch_number = 0

    # Vector of [valence_mean, arousal_mean, valence_std, arousal_std] for both
    song_id_vec = tf.zeros(shape=(0,), dtype=tf.int32)
    pred_vec = tf.zeros(shape=(0, 4), dtype=tf.float32)
    label_vec = tf.zeros(shape=(0, 4), dtype=tf.float32)

    while True:
      try:
        song_id, _, batch_x, batch_label = next(test_data_iter)
        # with tf.device("/GPU:0"):
        #   pred = model(batch_x, training=False)
        pred = model.predict(batch_x)
        # print(f"Pred shape: {pred.shape}")
        # print(f"Pred value: {pred}")

        ## Extracting data
        song_id_vec = tf.concat([song_id_vec, song_id], axis=0)
        pred_vec = tf.concat([pred_vec, pred], axis=0)
        label_vec = tf.concat([label_vec, batch_label], axis=0)

        batch_number += 1
      except:
        break

    return song_id_vec, label_vec, pred_vec

  mixed_song_id_vec, mixed_label_vec, mixed_pred_vec = infer(test_batch_iter, model_mixed)
  sep_song_id_vec, sep_label_vec, sep_pred_vec = infer(test_batch_sep_iter, model_sep)
  song_id_vec = tf.cast(mixed_song_id_vec[..., tf.newaxis], tf.float32)
  compiled_data = tf.concat([song_id_vec, mixed_label_vec, mixed_pred_vec, sep_pred_vec], axis=-1)
  # print(compiled_data)
  result_df = pd.DataFrame(compiled_data.numpy(), columns=columns)
  result_df.to_csv(csv_path_result, index=False)

  # A partilular row's stats
  row = result_df.iloc[0]
  gt_stats: tf.Tensor = tf.convert_to_tensor(row.values[1:5])
  mixed_stats: tf.Tensor = tf.convert_to_tensor(row.values[5:9])
  sep_stats: tf.Tensor = tf.convert_to_tensor(row.values[9:13])

  # All stats
  all_song_id = result_df.iloc[:, 0]
  gt_all_stats = tf.convert_to_tensor(result_df.iloc[:, 1:5], dtype=tf.float32)
  mix_all_stats = tf.convert_to_tensor(result_df.iloc[:, 5:9], dtype=tf.float32)
  sep_all_stats = tf.convert_to_tensor(result_df.iloc[:, 9:13], dtype=tf.float32)

  kl_all_mixed = compute_all_kl_divergence(gt_all_stats, mix_all_stats)
  kl_all_sep = compute_all_kl_divergence(gt_all_stats, sep_all_stats)

  kl_data_path = f"./kl_results/kl_result_fold_{fold}.csv"

  all_song_id_tf = tf.convert_to_tensor(all_song_id, dtype=tf.float32)[..., tf.newaxis]
  kl_data = tf.concat([all_song_id_tf, kl_all_mixed[..., tf.newaxis], kl_all_sep[..., tf.newaxis]], axis=-1)
  df_kl_data = pd.DataFrame(kl_data, columns=["song_id", "kl_mixed", "kl_sep"])
  df_kl_data.to_csv(kl_data_path, index=False)



