#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 13, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Abstract class for model architectures
"""

import tensorflow as tf
from bunch import Bunch


class BaseModel:
    def __init__(self, config: Bunch) -> None:
        self.config: Bunch = config

    def save(self):
        print("Saving model...")
        self.saver.save(self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(latest_checkpoint)
            print("Model loaded")

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
