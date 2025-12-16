import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold 


def data_preprocess(df, data_dir):
    data_list = []
    unique_paths = df['image_path'].unique()
    target_names = df['target_name'].unique()
    for paths in unique_paths:
        data = df[ df['image_path'] == paths ]
        row = { 'image_path': paths }
        for tar in target_names:
            row[tar] = data[ data['target_name'] == tar ]['target'].values[0]
        data_list.append(row)
    return pd.DataFrame(data_list)


class BiomassDS(Dataset):
    """ Dataset for single stream model"""
    def __init__(self, df, data_dir, transform, target_names):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.target_names = target_names
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        targets = torch.tensor([row[col] for col in self.target_names], dtype=torch.float32)
        return image, targets 

def create_strata(df, target_col, bins, n_folds, random_state=8964):
    kfolds = []
    dfcopy = df.copy()
    strata_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state) 
    dfcopy['strata'] = pd.cut(dfcopy[target_col], bins=bins, labels=False) 
    for train_idx, val_idx in strata_kfold.split(dfcopy, dfcopy['strata']):
        kfolds.append((train_idx, val_idx))
    return kfolds
