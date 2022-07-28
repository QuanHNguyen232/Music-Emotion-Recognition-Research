# %%

import tensorflow as tf
from tensorflow.python.keras import layers as L
from tensorflow.python.keras.models import Model

# %%

input_tensor = L.Input(shape=(128, 10000))
tensor = L.LSTM(256)(input_tensor)
tensor = L.Dropout(0.2)(tensor)
tensor = L.Dense(64, activation="relu")(tensor)
tensor = L.Dense(32, activation="relu")(tensor)
tensor = L.Dropout(0.2)(tensor)
tensor = L.Dense(8, activation="relu")(tensor)
tensor = L.Dense(1, activation="relu")(tensor)
model = Model(inputs=input_tensor, outputs=tensor)
model.summary()


