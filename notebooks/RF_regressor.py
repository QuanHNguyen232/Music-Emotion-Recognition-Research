#%%
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from mer.utils.const import get_config_from_json, setup_global_config
from mer.utils.utils import load_metadata, split_train_test

# Argument parsing
config_path = "../configs/config.json"
config = get_config_from_json(config_path)

##### Workaround to setup global config ############
setup_global_config(config, verbose=True)
from mer.utils.const import GLOBAL_CONFIG
##### End of Workaround #####

#%%
df = load_metadata(GLOBAL_CONFIG.ANNOTATION_SONG_LEVEL)
df

#%%
feat_data_dir = '../data/PMEmo/PMEmo2019/PMEmo2019/features'
mixed_csv_path = os.path.join(feat_data_dir, 'mixed_wav_static_feat.csv')
sep_csv_paths = [os.path.join(feat_data_dir, 'sep_bass_static_feat.csv'),
                os.path.join(feat_data_dir, 'sep_drums_static_feat.csv'),
                os.path.join(feat_data_dir, 'sep_other_static_feat.csv'),
                os.path.join(feat_data_dir, 'sep_vocals_static_feat.csv')]
# Get Mixed dataset
mix_df = pd.read_csv(mixed_csv_path)
mix_df.musicID = mix_df.musicID.astype(np.int64)
mix_df.rename(columns={'musicID': 'musicId'}, inplace=True)
mix_df_ = mix_df.merge(df, how='right', on='musicId')

# Get separated dataset
sep_df = pd.read_csv(sep_csv_paths[0])
sep_df.musicID = sep_df.musicID.astype(np.int64)
sep_df.rename(columns={'musicID': 'musicId'}, inplace=True)

for i in range(1, len(sep_csv_paths)):
    local_df = pd.read_csv(sep_csv_paths[i])
    local_df.drop(columns=['musicID'], inplace=True)
    sep_df = pd.concat([sep_df, local_df], axis=1)

sep_df_ = sep_df.merge(df, how='right', on='musicId')

# Print shape
mix_df.shape, mix_df_.shape, sep_df.shape, sep_df_.shape

#%% MIXED DATA

train_df, test_df = split_train_test(mix_df_, GLOBAL_CONFIG.TRAIN_RATIO)

train_df.shape, test_df.shape

#%%
labels = ['Arousal(mean)', 'Valence(mean)', 'Arousal(std)', 'Valence(std)']
x_train_df = train_df.drop(columns=labels+['musicId'])
y_train_df = train_df[labels]
x_test_df = test_df.drop(columns=labels+['musicId'])
y_test_df = test_df[['musicId'] + labels]

print(x_train_df.shape, y_train_df.shape, x_test_df.shape, y_test_df.shape)

x_train_df.head()

#%%
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(x_train_df, y_train_df)
# 2m 30s

#%% Get FEAT_IMPORTANCE
feat_importances = zip(x_train_df.columns, rf_reg.feature_importances_)
most_important = sorted(feat_importances, key=lambda x: x[1], reverse=True)
most_important[:10]

#%%
music_id = y_test_df.musicId.to_numpy()
music_id = np.expand_dims(music_id, axis=1)

pred = rf_reg.predict(x_test_df)
print(pred.shape)
pred = np.concatenate([pred, music_id], axis=1) # add id into predict to save as csv
print(pred.shape)

#%%
result_df = pd.DataFrame(pred, columns=labels+['musicId'])

result_df_ = result_df[['musicId'] + labels]
result_df_.musicId = result_df_.musicId.astype(np.int64)

result_df_

#%%
y_test_df

#%% SAVE
result_df_.to_csv('rf_mixed_all-feats_result.csv', index=False)

#%% SEP DATA

train_sep_df, test_sep_df = split_train_test(sep_df_, GLOBAL_CONFIG.TRAIN_RATIO)

train_sep_df.shape, test_sep_df.shape

#%% GET TRAIN/TEST
labels = ['Arousal(mean)', 'Valence(mean)', 'Arousal(std)', 'Valence(std)']
x_train_sep_df = train_sep_df.drop(columns=labels+['musicId'])
y_train_sep_df = train_sep_df[labels]
x_test_sep_df = test_sep_df.drop(columns=labels+['musicId'])
y_test_sep_df = test_sep_df[['musicId'] + labels]

print(x_train_sep_df.shape, y_train_sep_df.shape, x_test_sep_df.shape, y_test_sep_df.shape)

x_train_sep_df.head()

#%% TRAIN RF (FIT)
rf_reg_sep = RandomForestRegressor(random_state=42)
rf_reg_sep.fit(x_train_sep_df, y_train_sep_df)
# 10m 40s

#%% Get FEAT_IMPORTANCE
feat_importances_sep = zip(x_train_sep_df.columns, rf_reg_sep.feature_importances_)
most_important_sep = sorted(feat_importances_sep, key=lambda x: x[1], reverse=True)
most_important_sep[:10]

#%%
music_id = y_test_sep_df.musicId.to_numpy()
music_id = np.expand_dims(music_id, axis=1)

pred_sep = rf_reg_sep.predict(x_test_sep_df)
print(pred_sep.shape)
pred_sep = np.concatenate([pred_sep, music_id], axis=1) # add id into predict to save as csv
print(pred_sep.shape)

#%% CONVERT NP -> PD

result_sep_df = pd.DataFrame(pred_sep, columns=labels+['musicId'])

# switch musicId column
result_sep_df_ = result_sep_df[['musicId'] + labels]
result_sep_df_.musicId = result_sep_df_.musicId.astype(np.int64)

result_sep_df_

#%%
y_test_sep_df

#%% SAVE
result_sep_df_.to_csv('rf_sep_all-feats_result.csv', index=False)

