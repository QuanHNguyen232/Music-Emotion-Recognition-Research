import tensorflow as tf
from tensorflow.python.keras import layers as L
from tensorflow.python.keras.models import Model

def get_rnn_model(input_shape=(128, 10000), verbose=False):
  input_tensor = L.Input(shape=input_shape)
  tensor = L.LSTM(256)(input_tensor)
  tensor = L.Dropout(0.2)(tensor)
  tensor = L.Dense(64, activation="relu")(tensor)
  tensor = L.Dense(32, activation="relu")(tensor)
  tensor = L.Dropout(0.2)(tensor)
  tensor = L.Dense(8, activation="relu")(tensor)
  out_tensor = L.Dense(1, activation="relu")(tensor)
  model = Model(inputs=input_tensor, outputs=out_tensor)
  if verbose:
    model.summary()
  model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
  return model


