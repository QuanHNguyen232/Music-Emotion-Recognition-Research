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

import tensorflow as tf

from ritnet.trainer.trainer import TrainerWithDistMatrix

print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import argparse

from ritnet.utils.config import get_config_from_json, setup_global_config

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
  
  config_path = "../configs/training_config/training_config_4.json"
  model_config_path = "../configs/model_config/simplenet.json"

  config = get_config_from_json(config_path)
  model_config = get_config_from_json(model_config_path)

  ##### Workaround to setup global config ############
  setup_global_config(config)
  from ritnet.utils.config import GLOBAL_CONFIG
  ##### End of Workaround #####

  # Because the generator and some classes are based on the
  # GLOBAL_CONFIG, we have to import them after we set the config
  from ritnet.trainer.trainer import Trainer
  from ritnet.model.model_builder import build_unet_model
  from ritnet.trainer.optimizer import get_optimizer_by_config
  from ritnet.dataloader.dataloader import train_generator, test_generator

  # Data loader
  train_dataset = None
  if "sl" in GLOBAL_CONFIG.loss.name:
    train_dataset = tf.data.Dataset.from_generator(
      train_generator,
      output_signature=(
        tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.channel), dtype=tf.float32),
        tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.n_class), dtype=tf.float32),
        tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.n_class), dtype=tf.float32)
      )
    )
  else:
    train_dataset = tf.data.Dataset.from_generator(
      train_generator,
      output_signature=(
        tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.channel), dtype=tf.float32),
        tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.n_class), dtype=tf.float32),
      )
    )
  train_batch_dataset = train_dataset.batch(GLOBAL_CONFIG.batch_size)
  train_batch_iter = iter(train_batch_dataset)

  test_dataset = None
  if "sl" in GLOBAL_CONFIG.loss.name:
    test_dataset = tf.data.Dataset.from_generator(
      test_generator,
      output_signature=(
        tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.channel), dtype=tf.float32),
        tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.n_class), dtype=tf.float32),
        tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.n_class), dtype=tf.float32)
      )
    )
  else:
    test_dataset = tf.data.Dataset.from_generator(
      test_generator,
      output_signature=(
        tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.channel), dtype=tf.float32),
        tf.TensorSpec(shape=(GLOBAL_CONFIG.image_size.height, GLOBAL_CONFIG.image_size.width, GLOBAL_CONFIG.n_class), dtype=tf.float32)
      )
    )
  test_batch_dataset = test_dataset.batch(GLOBAL_CONFIG.batch_size)
  test_batch_iter = iter(test_batch_dataset)

  # Build model!
  model = build_unet_model(config, model_config, verbose=True)

  # Training
  
  optimizer = get_optimizer_by_config(GLOBAL_CONFIG.optimizer)
  
  history_path = f"../history/{GLOBAL_CONFIG.name}_{model_config.model_name}.npy"
  weights_path = f"../models/{GLOBAL_CONFIG.name}_{model_config.model_name}/checkpoint"

  # get the loss function by loss config
  from ritnet.trainer.loss import get_loss_func_by_loss_config
  loss_func = get_loss_func_by_loss_config(GLOBAL_CONFIG.loss)
  trainer = None
  if "sl" in GLOBAL_CONFIG.loss.name:
    trainer = TrainerWithDistMatrix(
      model,
      train_batch_iter,
      test_batch_iter,
      optimizer,
      loss_func,
      epochs=GLOBAL_CONFIG.epochs,
      steps_per_epoch=GLOBAL_CONFIG.steps_per_epoch, # 34000 // 8 = 4250
      valid_step=GLOBAL_CONFIG.valid_step,
      history_path=history_path,
      weights_path=weights_path,
      save_history=True
    )
  else:
    trainer = Trainer(
      model,
      train_batch_iter,
      test_batch_iter,
      optimizer,
      loss_func,
      epochs=GLOBAL_CONFIG.epochs,
      steps_per_epoch=GLOBAL_CONFIG.steps_per_epoch, # 34000 // 8 = 4250
      valid_step=GLOBAL_CONFIG.valid_step,
      history_path=history_path,
      weights_path=weights_path,
      save_history=True
    )

  history = trainer.train()

if __name__ == '__main__':
  main()
