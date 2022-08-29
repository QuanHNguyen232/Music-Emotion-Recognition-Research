# %%

import pandas as pd
import os

##### Workaround to setup global config ############
from mer.utils.const import get_config_from_json, setup_global_config
# Argument parsing
config_path = "../configs/config.json"
config = get_config_from_json(config_path)
setup_global_config(config, verbose=True)
from mer.utils.const import GLOBAL_CONFIG
##### End of Workaround #####

from mer.utils.utils import load_metadata, split_train_test

N_FOLD = 10

K_FOLD_ANNOTATION_FOLDER = GLOBAL_CONFIG.K_FOLD_ANNOTATION_FOLDER

os.makedirs(K_FOLD_ANNOTATION_FOLDER, exist_ok=True)

for fold in range(N_FOLD):
  df = load_metadata(GLOBAL_CONFIG.ANNOTATION_SONG_LEVEL)
  df = df.sample(frac=1).reset_index(drop=True)
  annotation_path = os.path.join(K_FOLD_ANNOTATION_FOLDER, str(fold) + ".csv")
  df.to_csv(annotation_path, index=False)
