# %%

import pandas as pd
import os

from mer.static_feat_extractor import mixed_feat_extractor, sep_feat_extractor

csv_path = "./result.csv"
result_df = pd.read_csv(csv_path)
result_df

# %%

# extract mixed .wav files
results = mixed_feat_extractor('../data/PMEmo/PMEmo2019/PMEmo2019/chorus_wav/')
results.to_csv('chorus_wav_static_feat.csv', index=False)

# extract separated .wav files
sep_types = ['bass', 'drums', 'other', 'vocals']    # result from pretrained Wave-U-Net
for sep_type in sep_types:
    results = sep_feat_extractor('../data/PMEmo/PMEmo2019/PMEmo2019/separation_16', sep_type)
    results.to_csv(f'sep_{sep_type}_static_feat.csv', index=False)


print('Done static_feat_extractor.py')


rf_mixed_path = "./rf_mixed_all-feats_result.csv"
rf_sep_path = "./rf_sep_all-feats_result.csv"
result_rf_path = "./result_rf.csv"

df_mix_data = pd.read_csv(rf_mixed_path)
df_sep_data = pd.read_csv(rf_sep_path)


new_df_mix = pd.concat([df_mix_data.iloc[:, 2], df_mix_data.iloc[:, 1], df_mix_data.iloc[:, 4], df_mix_data.iloc[:, 3]], axis=1)
new_df_sep = pd.concat([df_sep_data.iloc[:, 2], df_sep_data.iloc[:, 1], df_sep_data.iloc[:, 4], df_sep_data.iloc[:, 3]], axis=1)
gt_df = result_df.iloc[:, :5]
result_rf_df = pd.concat([gt_df, new_df_mix, new_df_sep], axis=1)


if not os.path.exists(result_rf_path):
  result_rf_df.to_csv(result_rf_path, index=False)
