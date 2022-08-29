#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 13, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

File contains trainer
"""

import os
import tensorflow as tf
import numpy as np
from typing import List

from .utils.const import GLOBAL_CONFIG

class Trainer():
  def __init__(self, 
      model, 
      training_batch_iter, 
      test_batch_iter,
      optimizer, 
      loss_function,
      epochs=1, 
      steps_per_epoch=20, 
      valid_step=5,
      history_path=None,
      weights_path=None,
      save_history=False) -> None:
    self.model = model
    self.training_batch_iter = training_batch_iter
    self.test_batch_iter = test_batch_iter
    self.optimizer = optimizer
    self.loss_function = loss_function
    self.epochs = epochs
    self.steps_per_epoch = steps_per_epoch
    self.valid_step = valid_step
    self.history_path = history_path
    self.weights_path = weights_path
    self.save_history = save_history
  
  def train_step(self, batch_x, batch_label, model, loss_function, optimizer, verbose: bool=False):
    with tf.device("/GPU:0"):
      with tf.GradientTape() as tape:
        logits = model(batch_x, training=True)
        loss = loss_function(batch_label, logits)
        if verbose:
          print(f"[Trainer][train_step()] logits shape: {logits.shape}")
          print(f"[Trainer][train_step()] batch_label shape: {batch_label.shape}")
      grads = tape.gradient(loss, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss

  def train(self, verbose: bool=False) -> List[np.ndarray]:
    if self.history_path != None and os.path.exists(self.history_path):
      # Sometimes, we have not created the files
      with open(self.history_path, "rb") as f:
        history = np.load(f, allow_pickle=True)
      epochs_loss, epochs_val_loss = history
      epochs_loss = epochs_loss.tolist()
      epochs_val_loss = epochs_val_loss.tolist()
    else:
      epochs_val_loss = []
      epochs_loss = []
    
    if self.weights_path != None and os.path.exists(self.weights_path + ".index"):
      try:
        self.model.load_weights(self.weights_path)
        print("Model weights loaded!")
      except:
        print("cannot load weights!")

    for epoch in range(self.epochs):
      losses = []

      with tf.device("/CPU:0"):
        step_pointer = 0
        while step_pointer < self.steps_per_epoch:
          # try:
          #   batch = next(self.training_batch_iter)
          # except:
          #   print()
          #   continue
          _, batch_x, batch_label = next(self.training_batch_iter)
          loss = self.train_step(batch_x, batch_label, self.model, self.loss_function, self.optimizer, verbose=verbose)
          
          if (step_pointer) % self.valid_step == 0:
            # print(
            #   "Training loss (for one batch) at step %d: %.4f"
            #   % (step_pointer, float(loss))
            # )
            # perform validation
            # try:
            #   val_batch = next(self.test_batch_iter)
            # except:
            #   continue
            val_batch = next(self.test_batch_iter)
            logits = self.model(val_batch[2], training=False)
            val_loss = self.loss_function(val_batch[3], logits)
            losses.append(loss)
            epochs_val_loss.append(val_loss)
            # print(f"exmaple logits: {logits}")
            print(f"Epoch {epoch} - Step {step_pointer} - Loss: {loss} - Validation loss: {val_loss}")
          # if (step_pointer) == self.steps_per_epoch:
          #   val_batch = next(self.test_batch_iter)
          #   logits = self.model(val_batch[1], training=False)
          #   val_loss = self.loss_function(val_batch[2], logits)
          #   epochs_val_loss.append(val_loss)

          step_pointer += 1
      epochs_loss.append(losses)

      # Save history and model
      if self.history_path != None and self.save_history:
        np.save(self.history_path, [epochs_loss, epochs_val_loss])
      
      if self.weights_path != None:
        self.model.save_weights(self.weights_path)
    
    # return history
    return [epochs_loss, epochs_val_loss]


class TrainerWithWave():
  def __init__(self, 
      model, 
      training_batch_iter, 
      test_batch_iter,
      optimizer, 
      loss_function,
      epochs=1, 
      steps_per_epoch=20, 
      valid_step=5,
      history_path=None,
      weights_path=None,
      save_history=False) -> None:
    self.model = model
    self.training_batch_iter = training_batch_iter
    self.test_batch_iter = test_batch_iter
    self.optimizer = optimizer
    self.loss_function = loss_function
    self.epochs = epochs
    self.steps_per_epoch = steps_per_epoch
    self.valid_step = valid_step
    self.history_path = history_path
    self.weights_path = weights_path
    self.save_history = save_history
  
  def train_step(self, batch_x, batch_label, model, loss_function, optimizer, verbose: bool=False):
    with tf.device("/GPU:0"):
      with tf.GradientTape() as tape:
        logits = model(batch_x, training=True)
        loss = loss_function(batch_label, logits)
        if verbose:
          print(f"[Trainer][train_step()] logits shape: {logits.shape}")
          print(f"[Trainer][train_step()] batch_label shape: {batch_label.shape}")
      grads = tape.gradient(loss, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss

  def train(self, verbose: bool=False) -> List[np.ndarray]:
    if self.history_path != None and os.path.exists(self.history_path):
      # Sometimes, we have not created the files
      with open(self.history_path, "rb") as f:
        history = np.load(f, allow_pickle=True)
      epochs_loss, epochs_val_loss = history
      epochs_loss = epochs_loss.tolist()
      epochs_val_loss = epochs_val_loss.tolist()
    else:
      epochs_val_loss = []
      epochs_loss = []
    
    if self.weights_path != None and os.path.exists(self.weights_path + ".index"):
      try:
        self.model.load_weights(self.weights_path)
        print("Model weights loaded!")
      except:
        print("cannot load weights!")

    for epoch in range(self.epochs):
      losses = []

      with tf.device("/CPU:0"):
        step_pointer = 0
        while step_pointer < self.steps_per_epoch:
          # try:
          #   batch = next(self.training_batch_iter)
          # except:
          #   print()
          #   continue
          batch_x, _, batch_label = next(self.training_batch_iter)
          loss = self.train_step(batch_x, batch_label, self.model, self.loss_function, self.optimizer, verbose=verbose)
          print(f"Epoch {epoch + 1} - Step {step_pointer + 1} - Loss: {loss}")
          losses.append(loss)

          if (step_pointer + 1) % self.valid_step == 0:
            print(
              "Training loss (for one batch) at step %d: %.4f"
              % (step_pointer + 1, float(loss))
            )
            # perform validation
            # try:
            #   val_batch = next(self.test_batch_iter)
            # except:
            #   continue
            val_batch = next(self.test_batch_iter)
            logits = self.model(val_batch[0], training=False)
            val_loss = self.loss_function(val_batch[2], logits)
            # print(f"exmaple logits: {logits}")
            print(f"Validation loss: {val_loss}\n-----------------")
          if (step_pointer + 1) == self.steps_per_epoch:
            val_batch = next(self.test_batch_iter)
            logits = self.model(val_batch[0], training=False)
            val_loss = self.loss_function(val_batch[2], logits)
            epochs_val_loss.append(val_loss)

          step_pointer += 1
      epochs_loss.append(losses)

      # Save history and model
      if self.history_path != None and self.save_history:
        np.save(self.history_path, [epochs_loss, epochs_val_loss])
      
      if self.weights_path != None:
        self.model.save_weights(self.weights_path)
    
    # return history
    return [epochs_loss, epochs_val_loss]