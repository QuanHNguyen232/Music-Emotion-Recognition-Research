#%%
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor

from mer.utils.const import get_config_from_json, setup_global_config
from mer.utils.utils import load_metadata, split_train_test

# Argument parsing
config_path = "../configs/config.json"
config = get_config_from_json(config_path)
feat_data_dir = '../data/PMEmo/PMEmo2019/PMEmo2019/features'
result_dir = './results'
kl_result_dir = "./kl_results/"

##### Workaround to setup global config ############
setup_global_config(config, verbose=True)
from mer.utils.const import GLOBAL_CONFIG
from mer.utils.utils import compute_all_kl_divergence
##### End of Workaround #####

#%%
rf = RandomForestRegressor()
print('rf.n_estimators', rf.n_estimators)
print('rf.criterion', rf.criterion)
print('rf.max_depth', rf.max_depth)
print('rf.min_samples_split',rf.min_samples_split)
print('rf.min_samples_leaf', rf.min_samples_leaf)
print('rf.min_weight_fraction_leaf', rf.min_weight_fraction_leaf)
print('rf.max_features', rf.max_features)
print('rf.max_leaf_nodes', rf.max_leaf_nodes)
print('rf.min_impurity_decrease', rf.min_impurity_decrease)
print('rf.max_samples', rf.max_samples)
#%%
def get_xy_train_test(df):
    train_df, test_df = split_train_test(df, GLOBAL_CONFIG.TRAIN_RATIO)

    labels = ['gt_valence_mean', 'gt_arousal_mean', 'gt_valence_std', 'gt_arousal_std']
    x_train_df = train_df.drop(columns=labels + ['song_id'])
    y_train_df = train_df[labels]
    x_test_df = test_df.drop(columns=labels + ['song_id'])
    y_test_df = test_df[['song_id'] + labels]

    return (x_train_df, y_train_df, x_test_df, y_test_df)

#%%
def train_mixed_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # prepare mix_df_
    mixed_csv_path = os.path.join(feat_data_dir, 'mixed_wav_static_feat.csv')
    mix_df = pd.read_csv(mixed_csv_path)
    mix_df.musicID = mix_df.musicID.astype(np.int64)
    mix_df.rename(columns={'musicID': 'song_id'}, inplace=True)
    mix_df_ = mix_df.merge(df, how='right', on='song_id')
    
    # train-test split    
    x_train_df, y_train_df, x_test_df, y_test_df = get_xy_train_test(mix_df_)
    
    # model fit -- 2m 30s
    rf_reg = RandomForestRegressor()
    rf_reg.fit(x_train_df, y_train_df)
    
    # predict
    y_hat = rf_reg.predict(x_test_df)
    mixed_result_df = pd.DataFrame(y_hat, columns=['mixed_valence_mean','mixed_arousal_mean', 'mixed_valence_std', 'mixed_arousal_std'])
    
    return mixed_result_df

#%%
def train_sep_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # prepare sep_df_
    sep_csv_paths = [os.path.join(feat_data_dir, 'sep_bass_static_feat.csv'),
                    os.path.join(feat_data_dir, 'sep_drums_static_feat.csv'),
                    os.path.join(feat_data_dir, 'sep_other_static_feat.csv'),
                    os.path.join(feat_data_dir, 'sep_vocals_static_feat.csv')]
    sep_df = pd.read_csv(sep_csv_paths[0])
    sep_df.musicID = sep_df.musicID.astype(np.int64)
    sep_df.rename(columns={'musicID': 'song_id'}, inplace=True)

    # concatnate all seps into 1 table (bass, drums, other, vocals)
    for i in range(1, len(sep_csv_paths)):
        local_df = pd.read_csv(sep_csv_paths[i])
        local_df.drop(columns=['musicID'], inplace=True)
        sep_df = pd.concat([sep_df, local_df], axis=1)

    sep_df_ = sep_df.merge(df, how='right', on='song_id')

    # train-test split
    x_train_df, y_train_df, x_test_df, y_test_df = get_xy_train_test(sep_df_)

    # model fit -- 10m 30s
    rf_reg = RandomForestRegressor()
    rf_reg.fit(x_train_df, y_train_df)

    # predict
    y_hat = rf_reg.predict(x_test_df)
    sep_result_df = pd.DataFrame(y_hat, columns=['sep_valence_mean','sep_arousal_mean', 'sep_valence_std', 'sep_arousal_std'])
    return sep_result_df

#%% Train by folds
for fold, file in enumerate(os.listdir(GLOBAL_CONFIG.K_FOLD_ANNOTATION_FOLDER)):
    print(f'Working on fold {fold}', end='\t')

    # load metadata
    df = pd.read_csv(os.path.join(GLOBAL_CONFIG.K_FOLD_ANNOTATION_FOLDER, file))
    new_col_name = {'musicId':          'song_id',
                    'Valence(mean)':    'gt_valence_mean',
                    'Arousal(mean)':    'gt_arousal_mean',
                    'Valence(std)':     'gt_valence_std',
                    'Arousal(std)':     'gt_arousal_std'
                    }
    df.rename(columns=new_col_name, inplace=True)
    df = df[['song_id', 'gt_valence_mean', 'gt_arousal_mean', 'gt_valence_std', 'gt_arousal_std']]  # swap col position to match with template

    # train RF models
    mixed_result_df = train_mixed_dataset(df)
    sep_result_df = train_sep_dataset(df)
    
    # save result    
    result_df_ = pd.concat([df, mixed_result_df, sep_result_df], axis=1)
    result_df_.to_csv(os.path.join(result_dir, f'rf_result_fold_{fold}.csv'), index=False)
    print(f'Saved fold {fold}')
    

#%% Calc KL Divergence
for fold, file in enumerate(os.listdir(result_dir)):
    if not file.startswith('rf'): continue

    result_df = pd.read_csv(os.path.join(result_dir, file))
    
    result_max_len = result_df.iloc[:, 5:6].dropna().shape[0]

    all_song_id = result_df.iloc[:result_max_len, 0]
    gt_all_stats = tf.convert_to_tensor(result_df.iloc[:result_max_len, 1:5], dtype=tf.float32)
    mix_all_stats = tf.convert_to_tensor(result_df.iloc[:result_max_len, 5:9], dtype=tf.float32)
    sep_all_stats = tf.convert_to_tensor(result_df.iloc[:result_max_len, 9:13], dtype=tf.float32)

    kl_all_mixed = compute_all_kl_divergence(gt_all_stats, mix_all_stats)
    kl_all_sep = compute_all_kl_divergence(gt_all_stats, sep_all_stats)
    
    all_song_id_tf = tf.convert_to_tensor(all_song_id, dtype=tf.float32)[..., tf.newaxis]
    kl_data = tf.concat([all_song_id_tf, kl_all_mixed[..., tf.newaxis], kl_all_sep[..., tf.newaxis]], axis=-1)
    df_kl_data = pd.DataFrame(kl_data.numpy(), columns=["song_id", "kl_mixed", "kl_sep"])
    df_kl_data.to_csv(os.path.join(kl_result_dir, f'kl_{file}'), index=False)

    print(f'Run {file}')

#%% Export Figs

