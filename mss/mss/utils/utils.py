#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 14, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Utils file
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple
import cv2
from PIL import Image


def show_img(img):
  plt.figure(facecolor='white')
  plt.axis("off")
  plt.imshow(img)
  plt.show()

def show_imgs(images: List[List[np.ndarray]], figsize: Tuple[int]=(10,10)) -> None:
  height, width = len(images), len(images[0])
  axes = []
  f = plt.figure(figsize=figsize, facecolor='white')
  for i in range(height * width):
    # Debug, plot figure
    axes.append(f.add_subplot(height, width, i + 1))
    # subplot_title=("Subplot"+str(""))
    # axes[-1].set_title(subplot_title)  
    plt.imshow(images[i // width][i % width])
  f.tight_layout()
  plt.show()

def preprocess_image(img: np.array, image_size=None) -> tf.Tensor:
  """_summary_

  Args:
    img (np.array): Expect image with shape (height, width, 3). With range [0, 255], int
    image_size (tuple, optional): Size of the image (height, width) (does not include the number of channel).
      Defaults to None.

  Returns:
    tf.Tensor: Tensor representation of the preprocessed image.
      Has shape (*image_size, img.shape[2]). Range [0, 1]
  """
  # Preprocessing image
  preprocessed_img = tf.cast(img, tf.float32)
  preprocessed_img /= 255.0
  assert tf.reduce_max(preprocessed_img) <= 1 and tf.reduce_min(preprocessed_img) >= 0, "Wrong behavior"
  if image_size != None:
    preprocessed_img = tf.image.resize(preprocessed_img, image_size, method=tf.image.ResizeMethod.BILINEAR)
  return preprocessed_img

def get_mask_by_layer(img: tf.Tensor, layer: int) -> tf.Tensor:
  """Utilitiy method for preprocess label

  Args:
    img (tf.Tensor): _description_
    layer (int): _description_

  Returns:
    tf.Tensor: _description_
  """
  return tf.cast(
    tf.math.argmax(img, axis=-1) == layer, # Get only the higest value layer
    tf.float32
  )[..., tf.newaxis] # expand last dimension

def preprocess_label(img: np.array, image_size=None) -> tf.Tensor:
  """_summary_

  Args:
    img (np.array): A numpy representation of the image. Expected range uint8 [0-255]

  Returns:
    tf.Tensor: Tensor representation of label
  """
  resized_img = img
  if image_size != None:
    resized_img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.BILINEAR)

  eye_mask = tf.cast(
    tf.reduce_any(tf.cast(resized_img, tf.bool), axis=-1),
    tf.float32
  )[..., tf.newaxis]

  background_mask = 1 - eye_mask

  # Masking the raw mask with the eye mask for each pupil, iris, and sclera
  pupil_mask = eye_mask * get_mask_by_layer(resized_img, 0) 

  iris_mask = eye_mask * get_mask_by_layer(resized_img, 1)

  sclera_mask = eye_mask * get_mask_by_layer(resized_img, 2)

  result = tf.concat(
    [background_mask, pupil_mask, iris_mask, sclera_mask],
    axis=-1
  )
  
  return result