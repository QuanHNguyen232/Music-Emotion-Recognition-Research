#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 14, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Example main file
"""
# %%

import os
from random import random
import sys
import collections
from typing import List, Dict, Tuple
import json
import csv
import pickle
from munch import Munch
import random

import tensorflow as tf

print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import tensorflow.python.keras.layers as layers
import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
import cv2
import tqdm

import argparse

from ritnet.trainer.trainer import Trainer
from ritnet.utils.utils import show_img, preprocess_image, preprocess_label, show_imgs
from ritnet.utils.config import get_config_from_json, setup_global_config
from ritnet.model.model_builder import build_unet_model

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

def main():
  
  # Argument parsing

  config_path = "../configs/training_config/training_config_2.json"
  config = get_config_from_json(config_path)
  ##### Workaround to setup global config ############
  setup_global_config(config)
  from ritnet.utils.config import GLOBAL_CONFIG
  ##### End of Workaround #####

  model_config_path = "../configs/model_config/simplenet.json"
  model_config = get_config_from_json(model_config_path)

  history_path = f"../history/{GLOBAL_CONFIG.name}_{model_config.model_name}.npy"
  history = np.load(history_path, allow_pickle=True)
  [epochs_loss, epochs_val_loss] = history

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

  plt.xlabel("Step")
  plt.ylabel("Loss")
  plt.legend()
  # plt.show()

  plt.savefig("history.png", transparent=False)
  
if __name__ == '__main__':
  main()







