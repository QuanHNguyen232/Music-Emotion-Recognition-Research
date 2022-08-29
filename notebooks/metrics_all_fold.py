# %%

import argparse
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd

import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
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
from tensorflow.python.keras import layers as L
from tensorflow.python.keras.models import Model

from mer.utils.const import get_config_from_json, setup_global_config

# Argument parsing
config_path = "../configs/config.json"
config = get_config_from_json(config_path)

##### Workaround to setup global config ############
setup_global_config(config, verbose=True)
from mer.utils.const import GLOBAL_CONFIG
##### End of Workaround #####

# %%



kl_data_path = "./kl_results/kl_result_fold_9.csv"
kl_data_folder = "./kl_results/"
kl_figs_folder = "figs"
os.makedirs(kl_figs_folder, exist_ok=True)
# Show only history gram
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'figure.figsize':(7,3.5), 'figure.dpi':100})
n_bins = 30
colors = ['blue', 'orange']
label = ["Mixed Data", "Separate Data"]
for fold, fname in enumerate(os.listdir(kl_data_folder)):

    df_kl_data = pd.read_csv(f"./kl_results/kl_result_fold_{fold}.csv")
    all_song_id = df_kl_data.iloc[:, 0]
    kl_all_mixed = df_kl_data.iloc[:, 1]
    kl_all_sep = df_kl_data.iloc[:, 2]
    kl_all_mix_sep = df_kl_data.iloc[:, 1:]

    # print(f"Mean KL Divergence of mixed source: {tf.reduce_mean(kl_all_mixed)}")
    # print(f"Mean KL Divergence of separated source: {tf.reduce_mean(kl_all_sep)}")
    # print(f"Sum KL Divergence of mixed source: {tf.reduce_sum(kl_all_mixed)}")
    # print(f"Sum KL Divergence of separated source: {tf.reduce_sum(kl_all_sep)}")

    # fig, axes = plt.subplots(nrows=2, ncols=5)

    
    # axes[0][0].hist(kl_all_mix_sep, n_bins, density=True, histtype='bar', color=colors, label=colors)
    # axes[0][0].legend(prop={'size': 10})
    # axes[0][0].set_title('KL Divergence Distribution Frequency Comparison')

    # plt.show()

    # Plot Histogram on x
    plt.hist(kl_all_mix_sep, bins=n_bins, histtype='bar', color=colors, label=label)
    plt.legend(prop={"size": 10})
    plt.gca().set(title='KL Divergence Distribution Comparison', ylabel='Frequency', xlabel="KL Divergence")
    # plt.show()

    plt.savefig(os.path.join(kl_figs_folder, f"kl_fold_{fold}.png"))
    plt.show()



# %%


# Show only continuous distribution
import numpy as np

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

# Create dataframe with values and probabilities
var_range = 40
probabilities, values = np.histogram(kl_all_mixed, bins=int(var_range), density=False)

# Plot probability distribution like in your example
df = pd.DataFrame(dict(prob=probabilities, value=values[:-1]))
# df.plot.line(x='value', y='prob')
plt.plot(df.value, df.prob, label="Mixed Data")
plt.gca().set(title='KL Divergence Distribution of training with both dataset', ylabel='Frequency', xlabel="KL Divergence")

probabilities1, values1 = np.histogram(kl_all_sep, bins=int(var_range), density=False)

df = pd.DataFrame(dict(prob=probabilities1, value=values1[:-1]))
plt.plot(df.value, df.prob, label="Separated Data")
# plt.gca().set(title='KL Divergence Distribution of Separated data', ylabel='Frequency', xlabel="KL Divergence")
plt.legend()
plt.show()





# show merged distribution

import seaborn as sns    # v 0.11.0
sns.histplot(data=kl_all_mixed, bins=30, alpha= 0.2, kde=True,
             edgecolor='white', linewidth=0.5,
             line_kws=dict(color='green', alpha=1,
                           linewidth=1.5, label='KDE_all_mixed'))

sns.histplot(data=kl_all_sep, bins=30, alpha= 0.2, kde=True,
             edgecolor='blue', linewidth=0.5,
             line_kws=dict(color='red', alpha=1,
                           linewidth=1.5, label='KDE_all_sep'))


plt.gca().get_lines()[0].set_color('black') # manually edit line color due to bug in sns v 0.11.0
plt.legend(frameon=False)

# TODO: density=True ?


