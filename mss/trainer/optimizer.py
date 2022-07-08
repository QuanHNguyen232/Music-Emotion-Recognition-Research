#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 28, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

File contains optimizers getter
"""

import tensorflow as tf

def get_optimizer_by_config(optimizer_config: str):
  optimizer = None
  if optimizer_config.name == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer_config.learning_rate)
  else:
    pass
  return optimizer