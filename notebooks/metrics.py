# %%

import argparse
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
%matplotlib inline
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
from mer.utils.utils import load_metadata, split_train_test, \
  preprocess_waveforms, get_spectrogram, plot_and_play, \
  pad_waveforms, plot_wave
from mer.model import get_rnn_model,get_rnn_model_2, Simple_CRNN_3
from mer.optimizer import get_Adam_optimizer, get_SGD_optimizer
from mer.loss import simple_mae_loss, simple_mse_loss
from mer.feature import load_wave_data, extract_spectrogram_features

filenames = tf.io.gfile.glob(str(GLOBAL_CONFIG.AUDIO_FOLDER) + '/*')
# Process with average annotation per song. 

df = load_metadata(GLOBAL_CONFIG.ANNOTATION_SONG_LEVEL)
df

# %%

train_df, test_df = split_train_test(df, GLOBAL_CONFIG.TRAIN_RATIO)

# %%

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

# %%

# Define both mode

model_name = "crnn_3"

# Model mixed
model_mixed = Simple_CRNN_3(input_shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL))
history_mixed_path = f"../history/{model_name}.npy"
weights_mixed_path = f"../models/{model_name}/checkpoint"
model_mixed.load_weights(weights_mixed_path)

# Model sep
# model_sep = Simple_CRNN_3(input_shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, GLOBAL_CONFIG.N_CHANNEL_SEP))
# history_sep_path = f"../history/{model_name}_sep.npy"
# weights_sep_path = f"../models/{model_name}_sep/checkpoint"
# model_sep.load_weights(weights_sep_path)



# %%

columns = [
  "song_id", "gt_valence_mean", "gt_arousal_mean",
  "gt_valence_std", "gt_arousal_std", "mixed_valence_mean", 
  "mixed_arousal_mean", "mixed_valence_std", "mixed_arousal_std",
  "sep_valence_mean", "sep_arousal_mean", "sep_valence_std",
  "sep_arousal_std"
]

# csv_path = "./result.csv" ## Uncomment this to process the reuslt from the cnn model
csv_path = "./result_rf.csv" ## Uncomment this to process the result from the random forest model

# method that use a model to infer and return prediction data
def infer(test_data_iter, model):
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

      print(f"Successfully inferred batch {batch_number}")
      batch_number += 1
    except:
      break

  return song_id_vec, label_vec, pred_vec

if not os.path.exists(csv_path):
  mixed_song_id_vec, mixed_label_vec, mixed_pred_vec = infer(test_batch_iter, model_mixed)
  sep_song_id_vec, sep_label_vec, sep_pred_vec = infer(test_batch_sep_iter, model_sep)
  # Export df
  song_id_vec = tf.cast(mixed_song_id_vec[..., tf.newaxis], tf.float32)
  compiled_data = tf.concat([song_id_vec, mixed_label_vec, mixed_pred_vec, sep_pred_vec], axis=-1)
  result_df = pd.DataFrame(compiled_data.numpy(), columns=columns)
  result_df.to_csv(csv_path, index=False)
else:
  # and load code
  result_df = pd.read_csv(csv_path)
  result_df

result_df

# %%

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

# %%

def compute_kl_divergence(gt_stats: tf.Tensor, pred_stats: tf.Tensor) -> tf.float32:
  """ Compute the KL Divergence of the two distribution based on their univariates
  KL Divergence applies in n-dimension with the formula being proven here:
    https://statproofbook.github.io/P/mvn-kl.html
  In this context we refer 1 to pred and 2 to ground truth. E.g: E1 is the covariance matrix of the prediction

  Args:
    gt_stats (tf.Tensor): [gt_valence_mean, gt_arousal_mean, gt_valence_std, gt_arousal_std]
    pred_stats (tf.Tensor): [pred_valence_mean, pred_arousal_mean, pred_valence_std, pred_arousal_std]

  Returns:
    tf.float32:the KL Divergence of the two distribution
  """
  assert gt_stats.shape == (4,), "Error in gt_stats shape"
  assert pred_stats.shape == (4,), "Error in pred_stats shape"

  [pred_valence_mean, pred_arousal_mean, pred_valence_std, pred_arousal_std] = pred_stats
  [gt_valence_mean, gt_arousal_mean, gt_valence_std, gt_arousal_std] = gt_stats

  pred_corvariance_mat = tf.convert_to_tensor([[tf.square(pred_valence_std), 0], [0, tf.square(pred_arousal_std)]])
  gt_corvariance_mat = tf.convert_to_tensor([[tf.square(gt_valence_std), 0], [0, tf.square(gt_arousal_std)]])

  gt_corvariance_mat_inv = tf.linalg.inv(gt_corvariance_mat)

  pred_mean = tf.convert_to_tensor([[pred_valence_mean, pred_arousal_mean]])
  gt_mean = tf.convert_to_tensor([[gt_valence_mean, gt_arousal_mean]])

  kl_divergence = 0.5 * (
    (gt_mean - pred_mean) @ gt_corvariance_mat_inv \
      @ tf.transpose(gt_mean - pred_mean, perm=[1, 0]) + \
        tf.linalg.trace(gt_corvariance_mat_inv @ pred_corvariance_mat) - \
          tf.math.log(tf.linalg.det(pred_corvariance_mat) / tf.linalg.det(gt_corvariance_mat)) - 2
  )

  return tf.squeeze(kl_divergence)

def compute_all_kl_divergence(gt_stats: tf.Tensor, pred_stats: tf.Tensor) -> tf.float32:
  """ Compute the KL Divergence of the two distribution based on their univariates
  KL Divergence applies in n-dimension with the formula being proven here:
    https://statproofbook.github.io/P/mvn-kl.html
  In this context we refer 1 to pred and 2 to ground truth. E.g: E1 is the covariance matrix of the prediction

  Args:
    gt_stats (tf.Tensor): (n_songs, 4) with 4: [gt_valence_mean, gt_arousal_mean, gt_valence_std, gt_arousal_std]
    pred_stats (tf.Tensor): (n_songs, 4) with 4: [pred_valence_mean, pred_arousal_mean, pred_valence_std, pred_arousal_std]

  Returns:
    tf.float32:the KL Divergence of the two distribution
  """

  pred_valence_mean, pred_arousal_mean, pred_valence_std, pred_arousal_std = pred_stats[:,0], pred_stats[:,1], pred_stats[:,2], pred_stats[:,3]
  gt_valence_mean, gt_arousal_mean, gt_valence_std, gt_arousal_std = gt_stats[:,0], gt_stats[:,1], gt_stats[:,2], gt_stats[:,3]

  pred_corvariance_mat = tf.concat([
    tf.concat([tf.square(pred_valence_std[..., tf.newaxis]), tf.zeros_like(pred_valence_std[..., tf.newaxis])], axis=-1)[..., tf.newaxis, :],
    tf.concat([tf.zeros_like(pred_arousal_std[..., tf.newaxis]), tf.square(pred_arousal_std[..., tf.newaxis])], axis=-1)[..., tf.newaxis, :]
  ], axis=-2)

  gt_corvariance_mat = tf.concat([
    tf.concat([tf.square(gt_valence_std[..., tf.newaxis]), tf.zeros_like(gt_valence_std[..., tf.newaxis])], axis=-1)[..., tf.newaxis, :],
    tf.concat([tf.zeros_like(gt_arousal_std[..., tf.newaxis]), tf.square(gt_arousal_std[..., tf.newaxis])], axis=-1)[..., tf.newaxis, :]
  ], axis=-2)

  gt_corvariance_mat_inv = tf.linalg.inv(gt_corvariance_mat)

  pred_mean = tf.concat([pred_valence_mean[..., tf.newaxis], pred_arousal_mean[..., tf.newaxis]], axis=-1)[..., tf.newaxis, :]
  gt_mean = tf.concat([gt_valence_mean[..., tf.newaxis], gt_arousal_mean[..., tf.newaxis]], axis=-1)[..., tf.newaxis, :]

  constant_shape = (gt_stats.shape[0], 1, 1)

  kl_divergence = 0.5 * (
    (gt_mean - pred_mean) @ gt_corvariance_mat_inv \
      @ tf.transpose(gt_mean - pred_mean, perm=[0, 2, 1]) + \
        tf.linalg.trace(gt_corvariance_mat_inv @ pred_corvariance_mat)[..., tf.newaxis, tf.newaxis] - \
          tf.math.log(tf.linalg.det(pred_corvariance_mat) / tf.linalg.det(gt_corvariance_mat))[..., tf.newaxis, tf.newaxis] - \
            tf.broadcast_to(2.0, shape=constant_shape)
  )
  
  return tf.squeeze(kl_divergence)

# %%

# kl_mixed = compute_kl_divergence(gt_stats, mixed_stats)
# kl_sep = compute_kl_divergence(gt_stats, sep_stats)
# print(f"Ground truth stats: {gt_stats}")
# print(f"Mixed source prediction stats: {mixed_stats}")
# print(f"Sep source prediction stats: {sep_stats}")
# print(f"KL-Divergence of mixed vs ground truth: {kl_mixed}")
# print(f"KL-Divergence of sep vs ground truth: {kl_sep}")


kl_all_mixed = compute_all_kl_divergence(gt_all_stats, mix_all_stats)
kl_all_sep = compute_all_kl_divergence(gt_all_stats, sep_all_stats)


# %%

print(f"Mean KL Divergence of mixed source: {tf.reduce_mean(kl_all_mixed)}")
print(f"Mean KL Divergence of separated source: {tf.reduce_mean(kl_all_sep)}")

print(f"Sum KL Divergence of mixed source: {tf.reduce_sum(kl_all_mixed)}")
print(f"Sum KL Divergence of separated source: {tf.reduce_sum(kl_all_sep)}")

# %%

# kl_data_path = "./kl_result.csv"
kl_data_path = "./kl_rf_result.csv"

# Export all kl data
if not os.path.exists(kl_data_path):
  all_song_id_tf = tf.convert_to_tensor(all_song_id, dtype=tf.float32)[..., tf.newaxis]
  kl_data = tf.concat([all_song_id_tf, kl_all_mixed[..., tf.newaxis], kl_all_sep[..., tf.newaxis]], axis=-1)
  df_kl_data = pd.DataFrame(kl_data, columns=["song_id", "kl_mixed", "kl_sep"])
  df_kl_data.to_csv(kl_data_path, index=False)
else:
  # and load code
  df_kl_data = pd.read_csv(kl_data_path)
  
df_kl_data


# %%

kl_all_mixed = df_kl_data.iloc[:, 1]
kl_all_sep = df_kl_data.iloc[:, 2]

# %%

"""
All plotting methods is taken and editted from:
  https://stackoverflow.com/questions/58989973/how-to-smooth-a-probability-distribution-plot-in-python
"""

# Show only history gram
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'figure.figsize':(7,3.5), 'figure.dpi':100})

# Plot Histogram on x
plt.hist(kl_all_mixed, bins=40)
plt.gca().set(title='KL Divergence Distribution of Mixed data', ylabel='Frequency', xlabel="KL Divergence")
plt.show()

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'figure.figsize':(7,3.5), 'figure.dpi':100})

# Plot Histogram on x
plt.hist(kl_all_sep, bins=40)
plt.gca().set(title='KL Divergence Distribution of Separated data', ylabel='Frequency', xlabel="KL Divergence")
plt.show()

# %%

# Show only continuous distribution
import numpy as np

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

# Create dataframe with values and probabilities
var_range = 40
probabilities, values = np.histogram(kl_all_mixed, bins=int(var_range), density=False)

# Plot probability distribution like in your example
df = pd.DataFrame(dict(prob=probabilities, value=values[:-1]))
# df.plot.line(x='value', y='prob')
plt.plot(df.value, df.prob, label="Mixed Data")
plt.gca().set(title='KL Divergence Distribution of training with both dataset', ylabel='Frequency', xlabel="KL Divergence")

probabilities1, values1 = np.histogram(kl_all_sep, bins=int(var_range), density=False)

df = pd.DataFrame(dict(prob=probabilities1, value=values1[:-1]))
plt.plot(df.value, df.prob, label="Separated Data")
# plt.gca().set(title='KL Divergence Distribution of Separated data', ylabel='Frequency', xlabel="KL Divergence")
plt.legend()
plt.show()

# %%

# show merged distribution

import seaborn as sns    # v 0.11.0
sns.histplot(data=kl_all_mixed, bins=30, alpha= 0.2, kde=True,
             edgecolor='white', linewidth=0.5,
             line_kws=dict(color='green', alpha=1,
                           linewidth=1.5, label='KDE_all_mixed'))

sns.histplot(data=kl_all_sep, bins=30, alpha= 0.2, kde=True,
             edgecolor='blue', linewidth=0.5,
             line_kws=dict(color='red', alpha=1,
                           linewidth=1.5, label='KDE_all_sep'))


plt.gca().get_lines()[0].set_color('black') # manually edit line color due to bug in sns v 0.11.0
plt.legend(frameon=False)

# TODO: density=True ?


