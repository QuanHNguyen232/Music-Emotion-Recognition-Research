# DEFAULT_FREQ = 44100
# DEFAULT_TIME = 45
# WAVE_ARRAY_LENGTH = DEFAULT_FREQ * DEFAULT_TIME

# WINDOW_TIME = 5
# WINDOW_SIZE = WINDOW_TIME * DEFAULT_FREQ

# TRAIN_RATIO = 0.8

# BATCH_SIZE = 16

# FREQUENCY_LENGTH = 129
# N_CHANNEL = 2
# SPECTROGRAM_TIME_LENGTH = 15502
# SPECTROGRAM_HALF_SECOND_LENGTH = 171
# SPECTROGRAM_5_SECOND_LENGTH = 1721
# MFCCS_TIME_LENGTH = 3876

# LEARNING_RATE = 1e-4

# SOUND_EXTENSION = ".wav"

# # The minimum second to be labeled in the dynamics files.
# MIN_TIME_END_POINT = 15

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

  # Setup
  GLOBAL_CONFIG.WAVE_ARRAY_LENGTH = GLOBAL_CONFIG.DEFAULT_FREQ * GLOBAL_CONFIG.DEFAULT_TIME
  GLOBAL_CONFIG.WINDOW_SIZE = GLOBAL_CONFIG.WINDOW_TIME * GLOBAL_CONFIG.DEFAULT_FREQ

  if verbose:
    print(GLOBAL_CONFIG)
  
################### End of Workaround #############
