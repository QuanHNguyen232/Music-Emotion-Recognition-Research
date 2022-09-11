# %%

import argparse
import os
from mer.utils.utils import compute_all_kl_divergence
import numpy as np
import librosa
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import scipy.stats as stats

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

# Export all kl results
for fold in range(10):
  result_file = f"./results/result_fold_{fold}.csv"
  kl_data_path = f"./kl_results/kl_result_fold_{fold}.csv"
  result_df = pd.read_csv(result_file)
  # All stats
  all_song_id = result_df.iloc[:, 0]
  gt_all_stats = tf.convert_to_tensor(result_df.iloc[:, 1:5], dtype=tf.float32)
  mix_all_stats = tf.convert_to_tensor(result_df.iloc[:, 5:9], dtype=tf.float32)
  sep_all_stats = tf.convert_to_tensor(result_df.iloc[:, 9:13], dtype=tf.float32)

  kl_all_mixed = compute_all_kl_divergence(gt_all_stats, mix_all_stats)
  kl_all_sep = compute_all_kl_divergence(gt_all_stats, sep_all_stats)

  all_song_id_tf = tf.convert_to_tensor(all_song_id, dtype=tf.float32)[..., tf.newaxis]
  kl_data = tf.concat([all_song_id_tf, kl_all_mixed[..., tf.newaxis], kl_all_sep[..., tf.newaxis]], axis=-1)
  df_kl_data = pd.DataFrame(kl_data, columns=["song_id", "kl_mixed", "kl_sep"])
  df_kl_data.to_csv(kl_data_path, index=False)

# Export all kl results
for fold in range(10):
  result_file = f"./results/rf_result_fold_{fold}.csv"
  kl_data_path = f"./kl_results/kl_rf_result_fold_{fold}.csv"
  result_df = pd.read_csv(result_file)
  # All stats
  all_song_id = result_df.iloc[:, 0]
  gt_all_stats = tf.convert_to_tensor(result_df.iloc[:, 1:5], dtype=tf.float32)
  mix_all_stats = tf.convert_to_tensor(result_df.iloc[:, 5:9], dtype=tf.float32)
  sep_all_stats = tf.convert_to_tensor(result_df.iloc[:, 9:13], dtype=tf.float32)

  kl_all_mixed = compute_all_kl_divergence(gt_all_stats, mix_all_stats)
  kl_all_sep = compute_all_kl_divergence(gt_all_stats, sep_all_stats)

  all_song_id_tf = tf.convert_to_tensor(all_song_id, dtype=tf.float32)[..., tf.newaxis]
  kl_data = tf.concat([all_song_id_tf, kl_all_mixed[..., tf.newaxis], kl_all_sep[..., tf.newaxis]], axis=-1)
  df_kl_data = pd.DataFrame(kl_data, columns=["song_id", "kl_mixed", "kl_sep"])
  df_kl_data.to_csv(kl_data_path, index=False)




# %%

# Save each kl results to separate images
# kl_data_path = "./kl_results/kl_result_fold_9.csv"
kl_data_folder = "./kl_results/"
kl_figs_folder = "figs"
os.makedirs(kl_figs_folder, exist_ok=True)
# Show only history gram
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'figure.figsize':(7,3.5), 'figure.dpi':100})
n_bins = 30
colors = ['blue', 'orange']
label = ["Mixed Data", "Separate Data"]
for fold in range(10):

  df_kl_data = pd.read_csv(f"./kl_results/kl_rf_result_fold_{fold}.csv")
  all_song_id = df_kl_data.iloc[:, 0]
  kl_all_mixed = df_kl_data.iloc[:, 1]
  kl_all_sep = df_kl_data.iloc[:, 2]
  kl_all_mix_sep = df_kl_data.iloc[:, 1:]

  # Plot Histogram on x
  plt.hist(kl_all_mix_sep, bins=n_bins, histtype='bar', color=colors, label=label)
  plt.legend(prop={"size": 10})
  plt.gca().set(title='KL Divergence Distribution Comparison', ylabel='Frequency', xlabel="KL Divergence")
  # plt.show()

  plt.savefig(os.path.join(kl_figs_folder, f"kl_rf_fold_{fold}.png"))
  plt.show()

# %%

# Plot all Save al kl results to an images

# kl_data_path = "./kl_results/kl_result_fold_9.csv"
# kl_data_folder = "./kl_results/"
kl_figs_folder = "figs"
os.makedirs(kl_figs_folder, exist_ok=True)
# Show only history gram
# plt.rcParams['figure.facecolor'] = 'white'
# plt.rcParams.update({'figure.figsize':(6,4), 'figure.dpi':100})
n_bins = 30
colors = ['blue', 'orange']
label = ["Mixed Data", "Separate Data"]

plt.figure()
fig, axes = plt.subplots(4, 5, figsize=(20, 10))

stats_crnn = []
stats_rf = []

def get_median(v):
  v = tf.reshape(v, [-1])
  m = v.get_shape()[0]//2
  return tf.reduce_min(tf.nn.top_k(v, m, sorted=False).values)

for fold in range(10):

  df_kl_data = pd.read_csv(f"./kl_results/kl_result_fold_{fold}.csv")
  all_song_id = df_kl_data.iloc[:, 0]
  kl_all_mixed = df_kl_data.iloc[:, 1]
  kl_all_sep = df_kl_data.iloc[:, 2]
  kl_all_mix_sep = df_kl_data.iloc[:, 1:]

  row = fold // 5
  col = fold % 5

  # compute stats
  mean_mixed = tf.reduce_mean(kl_all_mixed).numpy()
  mean_sep = tf.reduce_mean(kl_all_sep).numpy()
  med_mixed = get_median(kl_all_mixed).numpy()
  med_sep = get_median(kl_all_sep).numpy()
  _, p_value = stats.wilcoxon(kl_all_mixed.to_numpy(), y=kl_all_sep.to_numpy())
  stats_crnn.append([mean_mixed, mean_sep, med_mixed, med_sep, p_value])
  # Plot Histogram on x
  axes[row, col].hist(kl_all_mix_sep, bins=n_bins, histtype='bar', color=colors, label=label)
  # axes[row, col].legend(prop={"size": 10})
  # axes[row, col].gca().set(title='KL Divergence Distribution Comparison', ylabel='Frequency', xlabel="KL Divergence")
  # plt.show()

  # plt.savefig(os.path.join(kl_figs_folder, f"kl_rf_fold_{fold}.png"))

for fold in range(10):

  df_kl_data = pd.read_csv(f"./kl_results/kl_rf_result_fold_{fold}.csv")
  all_song_id = df_kl_data.iloc[:, 0]
  kl_all_mixed = df_kl_data.iloc[:, 1]
  kl_all_sep = df_kl_data.iloc[:, 2]
  kl_all_mix_sep = df_kl_data.iloc[:, 1:]

  # Compute stats
  mean_mixed = tf.reduce_mean(kl_all_mixed).numpy()
  mean_sep = tf.reduce_mean(kl_all_sep).numpy()
  med_mixed = get_median(kl_all_mixed).numpy()
  med_sep = get_median(kl_all_sep).numpy()
  _, p_value = stats.wilcoxon(kl_all_mixed.to_numpy(), y=kl_all_sep.to_numpy())
  stats_rf.append([mean_mixed, mean_sep, med_mixed, med_sep, p_value])

  row = (10 + fold) // 5
  col = (10 + fold) % 5

  # Plot Histogram on x
  axes[row, col].hist(kl_all_mix_sep, bins=n_bins, histtype='bar', color=colors, label=label)
  # axes[row, col].legend(prop={"size": 10})
  # axes[row, col].gca().set(title='KL Divergence Distribution Comparison', ylabel='Frequency', xlabel="KL Divergence")
  # plt.show()

  # plt.savefig(os.path.join(kl_figs_folder, f"kl_rf_fold_{fold}.png"))
    

# plt.legend(prop={"size": 10})
# plt.gca().set(title='KL Divergence Distribution Comparison', ylabel='Frequency', xlabel="KL Divergence")
plt.tight_layout()

plt.show()

# %%

# Export statistics of kl results data
stats_crnn_pd = pd.DataFrame(stats_crnn, columns=["mean_mixed", "mean_sep", "med_mixed", "med_sep", "wilcoxon"])
stats_crnn_pd.to_csv("./aggs/stats_crnn.csv", index=False)

stats_rf_pd = pd.DataFrame(stats_rf, columns=["mean_mixed", "mean_sep", "med_mixed", "med_sep", "wilcoxon"])
stats_rf_pd.to_csv("./aggs/stats_rf.csv", index=False)


# %%

# Testing wilcoxon stats.

test_kl_path = "./kl_results/kl_rf_result_fold_0.csv"
df_kl_data = pd.read_csv(test_kl_path)
all_song_id = df_kl_data.iloc[:, 0]
kl_all_mixed = df_kl_data.iloc[:, 1]
kl_all_sep = df_kl_data.iloc[:, 2]

# Based on kl_results data, perform a t-test (between model_mix and model sep) 
# for each fold each model type
 
# Creating data groups
data_group1 = kl_all_mixed.to_numpy()
data_group2 = kl_all_sep.to_numpy()
 
# Perform the two sample t-test with equal variances
# stats.ttest_ind(a=data_group1, b=data_group2, equal_var=True)

s, p = stats.wilcoxon(data_group1, y=data_group2)
p




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


