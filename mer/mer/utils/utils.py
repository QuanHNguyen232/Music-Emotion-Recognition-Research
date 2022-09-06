
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import sounddevice as sd

from .const import GLOBAL_CONFIG

def get_spectrogram(waveform, input_len=44100):
  """ Check out https://www.tensorflow.org/io/tutorials/audio

  Args:
      waveform ([type]): Expect waveform array of shape (>44100,)
      input_len (int, optional): [description]. Defaults to 44100.

  Returns:
      Tensor: Spectrogram of the 1D waveform. Shape (freq, time, 1)
  """
  max_zero_padding = min(input_len, tf.shape(waveform))
  # Zero-padding for an audio waveform with less than 44,100 samples.
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      (input_len - max_zero_padding),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plot_spectrogram(spectrogram, ax):
  """ Check out https://www.tensorflow.org/io/tutorials/audio

  Args:
      spectrogram ([type]): Expect shape (time step, frequency)
      ax (plt.axes[i]): [description]
  """
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

# def load_metadata(csv_folder):
#   """ Pandas load multiple csv file and concat them into one df.

#   Args:
#       csv_folder (str): Path to the csv folder

#   Returns:
#       pd.DataFrame: The concatnated one!
#   """
#   global_df = pd.DataFrame()
#   for i, fname in enumerate(os.listdir(csv_folder)):
#     # headers: song_id, valence_mean, valence_std, arousal_mean, arousal_std
#     df = pd.read_csv(os.path.join(csv_folder, fname), sep=r"\s*,\s*", engine="python")
#     global_df = pd.concat([global_df, df], axis=0)
  
#   # Reset the index
#   global_df = global_df.reset_index(drop=True)

#   return global_df

def load_metadata(csv_list, join_key: str="musicId"):
  """ Pandas load multiple csv file and concat them into one df.

  Args:
      csv_folder (str): Path to the csv folder

  Returns:
      pd.DataFrame: The concatnated one!
  """
  # headers: musicId, Arousal(mean), Valence(mean), Arousal(std), Valence(std)
  global_df = None
  for i, fname in enumerate(csv_list):
    df = pd.read_csv(fname, sep=r"\s*,\s*", engine="python")
    # global_df = pd.concat([global_df, df], axis=1)
    if global_df is None:
      global_df = df
    else:
      global_df = global_df.join(df.set_index(join_key), on=join_key, how="outer")
  
  # Reset the index
  global_df = global_df.reset_index(drop=True)

  return global_df

def split_train_test(df: pd.DataFrame, train_ratio: float):
  train_size = int(len(df) * train_ratio)
  train_df: pd.DataFrame = df[:train_size]
  train_df = train_df.reset_index(drop=True)
  test_df: pd.DataFrame = df[train_size:]
  test_df = test_df.reset_index(drop=True)
  return train_df, test_df

def plot_wave(waveforms, second_id = 0, second_length = 10, channel = 0):
  from_id = int(GLOBAL_CONFIG.DEFAULT_FREQ * second_id)
  to_id = min(int(GLOBAL_CONFIG.DEFAULT_FREQ * (second_id + second_length)), waveforms.shape[0])

  fig, axes = plt.subplots(1, figsize=(12, 4))
  timescale = np.arange(to_id - from_id)
  axes.plot(timescale, waveforms[from_id:to_id, channel].numpy())
  axes.set_title('Waveform')
  axes.set_xlim([0, int(GLOBAL_CONFIG.DEFAULT_FREQ * second_length)])
  plt.show()

def plot_and_play(test_audio, second_id = 24.0, second_length = 1, channel = 0):
  """ Plot and play

  Args:
      test_audio ([type]): [description]
      second_id (float, optional): [description]. Defaults to 24.0.
      second_length (int, optional): [description]. Defaults to 1.
      channel (int, optional): [description]. Defaults to 0.
  """
  # Spectrogram of one second
  from_id = int(GLOBAL_CONFIG.DEFAULT_FREQ * second_id)
  to_id = min(int(GLOBAL_CONFIG.DEFAULT_FREQ * (second_id + second_length)), test_audio.shape[0])

  test_spectrogram = get_spectrogram(test_audio[from_id:, channel], input_len=int(GLOBAL_CONFIG.DEFAULT_FREQ * second_length))
  print(test_spectrogram.shape)
  fig, axes = plt.subplots(2, figsize=(12, 8))
  timescale = np.arange(to_id - from_id)
  axes[0].plot(timescale, test_audio[from_id:to_id, channel].numpy())
  axes[0].set_title('Waveform')
  axes[0].set_xlim([0, int(GLOBAL_CONFIG.DEFAULT_FREQ * second_length)])

  plot_spectrogram(test_spectrogram.numpy(), axes[1])
  axes[1].set_title('Spectrogram')
  plt.show()

  # Play sound
  sd.play(test_audio[from_id: to_id, channel], blocking=True)

def pad_waveforms(waveforms, input_len):
  """ Get the first input_len value of the waveforms, if not exist, pad it with 0.

  Args:
      waveforms ([type]): [description]
      input_len ([type]): [description]

  Returns:
      [type]: [description]
  """
  n_channel = waveforms.shape[-1]
  preprocessed = np.zeros((input_len, n_channel))
  if input_len <= waveforms.shape[0]:
    preprocessed = waveforms[:input_len, :]
  else:
    preprocessed[:waveforms.shape[0], :] = waveforms
  preprocessed = tf.convert_to_tensor(preprocessed)
  return preprocessed

def preprocess_waveforms(waveforms):
  # preprocessed = (
  #   preprocessed - tf.broadcast_to(
  #     tf.expand_dims(
  #       tf.reduce_mean(preprocessed, axis=1), axis=1
  #     ), 
  #     preprocessed.shape
  #   )
  # ) / (tf.broadcast_to(tf.expand_dims(tf.math.reduce_std(preprocessed, axis=1), axis=1), preprocessed.shape) + 1e-8)
  preprocessed = waveforms / 2.0 + 0.5
  return preprocessed

def tanh_to_sigmoid(inputs):
  """ Convert from tanh range to sigmoid range

  Args:
    inputs (): number of np array of number

  Returns:
    number or array-like object: changed range object
  """
  return (inputs + 1.0) / 2.0

def get_CAM(model, img, actual_label, loss_func, layer_name='block5_conv3'):

  model_grad = tf.keras.Model(model.inputs, 
                      [model.get_layer(layer_name).output, model.output])
  
  with tf.GradientTape() as tape:
      conv_output_values, predictions = model_grad(img)

      # watch the conv_output_values
      tape.watch(conv_output_values)
      
      # Calculate loss as in the loss func
      try:
        loss, _ = loss_func(actual_label, predictions)
      except:
        loss = loss_func(actual_label, predictions)
      print(f"Loss: {loss}")
  
  # get the gradient of the loss with respect to the outputs of the last conv layer
  grads_values = tape.gradient(loss, conv_output_values)
  grads_values = tf.reduce_mean(grads_values, axis=(0,1,2))
  
  conv_output_values = np.squeeze(conv_output_values.numpy())
  grads_values = grads_values.numpy()
  
  # weight the convolution outputs with the computed gradients
  for i in range(conv_output_values.shape[-1]): 
      conv_output_values[:,:,i] *= grads_values[i]
  heatmap = np.mean(conv_output_values, axis=-1)
  
  heatmap = np.maximum(heatmap, 0)
  heatmap /= heatmap.max()
  
  del model_grad, conv_output_values, grads_values, loss
  
  return heatmap

# Statistics
def plot_history(history_path):

  import matplotlib.pyplot as plt
  # Plot
  with open(history_path, "rb") as f:
    [epochs_loss, epochs_val_loss] = np.load(f, allow_pickle=True)


  e_loss = [k[0] for k in epochs_loss]

  e_all_loss = []

  id = 0
  time_val = []
  for epoch in epochs_loss:
    for step in epoch:
      e_all_loss.append(step.numpy())
      id += 1
    time_val.append(id)

  plt.figure(facecolor='white')
  plt.plot(np.arange(0, len(e_all_loss), 1), e_all_loss, label = "train loss")
  plt.plot(time_val, epochs_val_loss, label = "val loss")

  # plt.plot(np.arange(1,len(e_loss)+ 1), e_loss, label = "train loss")
  # plt.plot(np.arange(1,len(epochs_val_loss)+ 1), epochs_val_loss, label = "val loss")
  plt.xlabel("Step")
  plt.ylabel("Loss")
  plt.legend()
  
  plt.show()

# Deprecated
def compute_kl_divergence(pred_stats: tf.Tensor, gt_stats: tf.Tensor) -> tf.float32:
  """ Compute the KL Divergence of the two distribution based on their univariates
  KL Divergence applies in n-dimension with the formula being proven here:
    https://statproofbook.github.io/P/mvn-kl.html
  In this context we refer 2 to pred and 1 to ground truth. E.g: E2 is the covariance matrix of the prediction
  Here we actually mistaken the pred and gt, so we did a quick swap of parameters when use (E.g: compute_all_kl_divergence(gt_stats, pred_stats)

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
  In this context we refer 1 to ground truth and 2 to pred. E.g: E2 is the covariance matrix of the prediction
  
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

  pred_corvariance_mat_inv = tf.linalg.inv(pred_corvariance_mat)

  pred_mean = tf.concat([pred_valence_mean[..., tf.newaxis], pred_arousal_mean[..., tf.newaxis]], axis=-1)[..., tf.newaxis, :]
  gt_mean = tf.concat([gt_valence_mean[..., tf.newaxis], gt_arousal_mean[..., tf.newaxis]], axis=-1)[..., tf.newaxis, :]

  constant_shape = (gt_stats.shape[0], 1, 1)

  kl_divergence = 0.5 * (
    (pred_mean - gt_mean) @ pred_corvariance_mat_inv \
      @ tf.transpose(pred_mean - gt_mean, perm=[0, 2, 1]) + \
        tf.linalg.trace(pred_corvariance_mat_inv @ gt_corvariance_mat)[..., tf.newaxis, tf.newaxis] - \
          tf.math.log(tf.linalg.det(gt_corvariance_mat) / tf.linalg.det(pred_corvariance_mat))[..., tf.newaxis, tf.newaxis] - \
            tf.broadcast_to(2.0, shape=constant_shape)
  )
  
  return tf.squeeze(kl_divergence)