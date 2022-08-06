#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 28, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

File contains loss functions
"""

import os
from munch import Munch
import tensorflow as tf
import numpy as np
from typing import List

def simple_mse_loss(true, pred):

  # loss_valence = tf.reduce_sum(tf.square(true[..., 0] - pred[0][..., 0])) / true.shape[0] # divide by batch size
  # loss_arousal = tf.reduce_sum(tf.square(true[..., 1] - pred[1][..., 0])) / true.shape[0] # divide by batch size

  # return loss_valence + loss_arousal

  loss = tf.reduce_sum(tf.square(true - pred)) / true.shape[0]

  return loss

def simple_mae_loss(true, pred):
  loss = tf.reduce_sum(tf.abs(true - pred)) / true.shape[0]
  return loss
