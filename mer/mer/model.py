import tensorflow as tf
from tensorflow.python.keras import layers as L
from tensorflow.python.keras.models import Model

from .utils.const import GLOBAL_CONFIG

def get_rnn_model(input_shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, 2), verbose=False):
  input_tensor = L.Input(shape=input_shape)
  tensor = L.Permute((2, 1, 3))(input_tensor)
  tensor = L.Dense(1, "relu")(tensor)
  tensor = tf.squeeze(tensor, axis=-1)
  # tensor = L.Resizing(GLOBAL_CONFIG.FREQUENCY_LENGTH, 1024)(tensor)
  tensor = L.LSTM(256)(tensor)
  tensor = L.Dropout(0.2)(tensor)
  tensor = L.Dense(64, activation="relu")(tensor)
  tensor = L.Dense(32, activation="relu")(tensor)
  tensor = L.Dropout(0.2)(tensor)
  tensor = L.Dense(8, activation="relu")(tensor)
  out_tensor = L.Dense(4, activation="relu")(tensor)
  model = Model(inputs=input_tensor, outputs=out_tensor)

  if verbose:
    model.summary()
  
  return model

  
  


