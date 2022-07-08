#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 14, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

Config json processor file
"""

import json
from munch import Munch

def get_config_from_json(json_file):
  """
  Get the config from a json file
  :param json_file:
  :return: config(namespace) or config(dictionary)
  """
  # parse the configurations from the config json file provided
  with open(json_file, 'r') as config_file:
    config_dict = json.load(config_file)

  # convert the dictionary to a namespace using bunch lib
  config = Munch.fromDict(config_dict)

  return config

################### Workaround of a GLOBAL_CONFIG ####################

GLOBAL_CONFIG: Munch = None

def setup_global_config(config: Munch, verbose: bool=False) -> None:
  global GLOBAL_CONFIG
  GLOBAL_CONFIG = config
  if verbose:
    print(GLOBAL_CONFIG)

################### End of Workaround #############

