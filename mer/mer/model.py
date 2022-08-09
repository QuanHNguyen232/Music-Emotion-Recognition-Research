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

def get_rnn_model_2(input_shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, 2), verbose=False):
  input_tensor = L.Input(shape=input_shape)
  tensor = L.Permute((2, 1))(input_tensor)
  # tensor = L.Dense(1, "relu")(tensor)
  # tensor = tf.squeeze(tensor, axis=-1)
  # tensor = L.Resizing(GLOBAL_CONFIG.FREQUENCY_LENGTH, 1024)(tensor)
  tensor = L.LSTM(512)(tensor)
  tensor = L.Dropout(0.2)(tensor)
  tensor = L.Dense(256, activation="relu")(tensor)
  tensor = L.Dense(256, activation="relu")(tensor)
  tensor = L.Dropout(0.2)(tensor)
  tensor = L.Dense(128, activation="relu")(tensor)
  tensor = L.Dense(128, activation="relu")(tensor)
  tensor = L.Dropout(0.2)(tensor)
  tensor = L.Dense(64, activation="relu")(tensor)
  tensor = L.Dense(64, activation="relu")(tensor)
  tensor = L.Dense(32, activation="relu")(tensor)
  tensor = L.Dense(32, activation="relu")(tensor)
  tensor = L.Dense(8, activation="relu")(tensor)
  tensor = L.Dense(8, activation="relu")(tensor)
  out_tensor = L.Dense(4)(tensor)
  model = Model(inputs=input_tensor, outputs=out_tensor)

  if verbose:
    model.summary()
  
  return model  
  
def Simple_CRNN_3(input_shape=(GLOBAL_CONFIG.SPECTROGRAM_TIME_LENGTH, GLOBAL_CONFIG.FREQUENCY_LENGTH, 1)):
  """ CRNN that uses GRU

  Args:
    inputs (tf.Tensor): Expect tensor shape (batch, width, height, channel)

  Returns:
    [type]: [description]
  """
  
  inputs = L.Input(shape=input_shape)
  tensor = L.Permute((2, 1, 3))(inputs)
  tensor = tf.keras.layers.Resizing(GLOBAL_CONFIG.FREQUENCY_LENGTH, 2048)(tensor)
  
  tensor = L.Conv2D(64, (5,5), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(64 // 2, (1,1), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(128, (5,5), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(128 // 2, (1,1), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(256, (5,5), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(256 // 2, (1,1), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(512, (5,5), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(512 // 2, (1,1), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(1024, (3, 3), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(1024 // 2, (1,1), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = tf.squeeze(tensor, axis=1)

  tensor = L.Permute((2, 1))(tensor)

  tensor = L.LSTM(256, activation="tanh", return_sequences=True)(tensor)
  tensor = L.LSTM(128, activation="tanh", return_sequences=True)(tensor)
  tensor = L.LSTM(64, activation="tanh")(tensor)
  tensor = L.Dense(512, activation="relu")(tensor)
  tensor = L.Dense(256, activation="relu")(tensor)
  tensor = L.Dense(64, activation="relu")(tensor)
  out = L.Dense(4)(tensor)

  # tensor_1 = L.Bidirectional(L.LSTM(128, return_sequences=True))(tensor)
  # tensor_1 = L.Bidirectional(L.LSTM(128, return_sequences=True))(tensor_1)
  # tensor_1 = L.Bidirectional(L.LSTM(128))(tensor_1)
  # tensor_1 = L.Dense(512, activation="relu")(tensor_1)
  # tensor_1 = L.Dense(256, activation="relu")(tensor_1)
  # tensor_1 = L.Dense(64, activation="relu")(tensor_1)
  # out_1 = L.Dense(1, activation="relu")(tensor_1)

  # tensor_2 = L.Bidirectional(L.LSTM(128, return_sequences=True))(tensor)
  # tensor_2 = L.Bidirectional(L.LSTM(128, return_sequences=True))(tensor_2)
  # tensor_2 = L.Bidirectional(L.LSTM(128))(tensor_2)
  # tensor_2 = L.Dense(512, activation="relu")(tensor_2)
  # tensor_2 = L.Dense(256, activation="relu")(tensor_2)
  # tensor_2 = L.Dense(64, activation="relu")(tensor_2)
  # out_2 = L.Dense(1, activation="relu")(tensor_2)
  

  model = Model(inputs=inputs, outputs=out)
  return model

