from fileinput import filename
import opensmile
import os
import pandas as pd

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def mixed_feat_extractor(dir_path: str) -> pd.DataFrame:
    ''' Extract features from mixed (original) wav files
    
    Arguments:
        dir_path -- directory path of wav files.

    Returns:
        results -- pd.DataFrame of shape (num_files, num_features) with num_features=6374
    
    Note: Folder structure of dir_path:
    .
    ├───1.mp3
    ├───4.mp3
    ├───5.mp3
    ├───6.mp3
    ...
    └───999.mp3
    '''
    count = 0
    results = pd.DataFrame()
    for file in os.listdir(dir_path):
        count+=1
        print('working on file', file, '\t\tDone:', count, 'files')
        musicId = file.split('.')[0]
        y = smile.process_file(os.path.join(dir_path, file))
        y.insert(0, 'musicID', [musicId])
        results = pd.concat([results, y])
    results.reset_index(drop=True, inplace=True)

    print(results.shape)
    
    return results

def sep_feat_extractor(dir_path: str, sep_type: str) -> pd.DataFrame:
    ''' Extract features from separated wav files
    
    Arguments:
        dir_path -- directory path of wav files (after separation with Wave-U-Net).

    Returns:
        results -- pd.DataFrame of shape (num_files, num_features) with num_features=6374
    
    Note: Folder structure of dir_path:
    .
    ├───1.mp3
    │       1.mp3_bass.wav
    │       1.mp3_drums.wav
    │       1.mp3_other.wav
    │       1.mp3_vocals.wav
    │
    ...
    │
    └───999.mp3
        999.mp3_bass.wav
        999.mp3_drums.wav
        999.mp3_other.wav
        999.mp3_vocals.wav
    '''
    count = 0
    results = pd.DataFrame()
    for file_dir in os.listdir(dir_path):
        musicId = file_dir.split('.')[0]
        file_path = os.path.join(dir_path, file_dir, f'{musicId}.mp3_{sep_type}.wav')
        y = smile.process_file(file_path)
        y.insert(0, 'musicID', [musicId])
        results = pd.concat([results, y])

        count+=1
        print('id= %5s \t Done: %3d files \t working on: %s'%(musicId, count, file_path))
    results.reset_index(drop=True, inplace=True)

    print(results.shape)
    
    return results

if __name__ == '__main__':

    # extract mixed .wav files
    results = mixed_feat_extractor('../data/PMEmo/PMEmo2019/PMEmo2019/chorus_wav/')
    results.to_csv('chorus_wav_static_feat.csv', index=False)

    # extract separated .wav files
    sep_types = ['bass', 'drums', 'other', 'vocals']    # result from pretrained Wave-U-Net
    for sep_type in sep_types:
        results = sep_feat_extractor('../data/PMEmo/PMEmo2019/PMEmo2019/separation_16', sep_type)
        results.to_csv(f'sep_{sep_type}_static_feat.csv', index=False)


    print('Done static_feat_extractor.py')