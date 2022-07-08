#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 13, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Abstract class for trainer algorithms
"""

from bunch import Bunch

from .base_model import BaseModel


class BaseTrainer:
    def __init__(self, model: BaseModel, data, config: Bunch, logger) -> None:
        self.model: BaseModel = model
        self.logger = logger
        self.config = config
        self.data = data

    def train(self):
        """Implement the logic of train:
    -loop over the number of iterations in the config and call the train step
    -add any summaries you want using the summary
    """
        raise NotImplementedError

    def train_epoch(self):
        """Implement the logic of epoch:
    -loop over the number of iterations in the config and call the train step
    -add any summaries you want using the summary
    """
        raise NotImplementedError

    def train_step(self):
        """Implement the logic of epoch:
    -loop over the number of iterations in the config and call the train step
    -add any summaries you want using the summary
    """
        raise NotImplementedError
