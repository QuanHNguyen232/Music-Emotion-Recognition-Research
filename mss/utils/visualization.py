#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 16, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

visualization file
"""

import skimage
import matplotlib as plt
import numpy as np
import tensorflow as tf

def draw_image_histogram(example_path: str) -> None:
  """_summary_

  https://datacarpentry.org/image-processing/05-creating-histograms/

  Args:
    example_path (str): _description_
  """
  # read original image, in full color
  image = skimage.io.imread(example_path)

  # display the image
  fig, ax = plt.subplots()
  plt.imshow(image)
  plt.show()

  # tuple to select colors of each channel line
  colors = ("red", "green", "blue")
  channel_ids = (0, 1, 2)

  # create the histogram plot, with three lines, one for
  # each color
  plt.figure()
  plt.xlim([0, 256])
  for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
      image[:, :, channel_id], bins=16, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=c)

  plt.title("Color Histogram")
  plt.xlabel("Color value")
  plt.ylabel("Pixel count")

  plt.show()


def examine_image(np_image: np.array) -> None:
  """Examine the image's distribution and information.

  Args:
    np_image (np.array): Input np array image, expect input to 
      be int type and ranged [0, 255].
  """
  print(f"Image shape: {np_image.shape}")
  total_size = np_image.shape[0] * np_image.shape[1]
  zero_c1 = tf.reduce_sum(tf.cast(np_image[:,:,0] == 0, tf.float32))
  zero_c2 = tf.reduce_sum(tf.cast(np_image[:,:,1] == 0, tf.float32))
  zero_c3 = tf.reduce_sum(tf.cast(np_image[:,:,2] == 0, tf.float32))
  print(f"The percentage of 0's pixel in channel 0, 1, 2 is {zero_c1 / total_size * 100}%, {zero_c2 / total_size * 100}%, {zero_c3 / total_size * 100}%,")
  print(f"Unique pixel value: {np.unique(np_image)}")
  # print(example_image[example_image[:,:,0] != 0])