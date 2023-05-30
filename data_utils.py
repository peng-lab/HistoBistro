from pathlib import Path
from typing import Iterable, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset


def get_cohort_df(clini_table: Path, slide_csv: Path, feature_dir: Path,
                    target_labels: Iterable[str], categories: Iterable[str], cohort: str, clini_info: dict = {}) -> pd.DataFrame:
    
    clini_df = pd.read_csv(clini_table, dtype=str) if Path(clini_table).suffix == '.csv' else pd.read_excel(
        clini_table, dtype=str)
    slide_df = pd.read_csv(slide_csv, dtype=str)
    df = clini_df.merge(slide_df, on='PATIENT')
    # adapt dataframe to case sensitive clini tables
    df = df.rename({
        'MSI': 'isMSIH',
        'Age': 'AGE'
    }, axis=1)

    # remove columns not in target_labels
    for key in df.columns:
        if key not in target_labels + ['PATIENT', 'SLIDE', 'FILENAME', *list(clini_info.keys())]:
            df.drop(key, axis=1, inplace=True)
    # remove rows/slides with non-valid labels
    for target in target_labels:
        df = df[df[target].isin(categories)]
    # remove slides we don't have
    h5s = set(feature_dir.glob('**/*.h5'))
    assert h5s, f'no features found in {feature_dir}!'
    h5_df = pd.DataFrame(h5s, columns=['slide_path'])
    # h5_df['FILENAME'] = h5_df.slide_path.map(lambda p: p.stem.split('.')[0])
    h5_df['FILENAME'] = h5_df.slide_path.map(lambda p: p.stem.split('_')[0].split('.')[0]) if cohort=='TCGA' else h5_df.slide_path.map(lambda p: p.stem)
    df = df.merge(h5_df, on='FILENAME')
    # reduce to one row per patient with list of slides in `df['slide_path']`
    patient_df = df.groupby('PATIENT').first().drop(columns='slide_path')
    patient_slides = df.groupby('PATIENT').slide_path.apply(list)
    df = patient_df.merge(patient_slides, left_on='PATIENT', right_index=True).reset_index()
        
    return df


def transform_clini_info(df: pd.DataFrame, label: str, mean: np.ndarray, std: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """ transform columns with categorical features to integers and normalize them with given mean and std dev"""
    # fill missing columns with 0
    if label not in df.keys():
        df[label] = 0
        return df, mean, std

    if label == 'AGE':
        # only choose rows with valid labels
        col = df.loc[df[label].str.isdigit().notnull(), label]
        # map columns to integers
        col = col.astype(int)
        # normalize columns
        if mean is None:
            mean = col.mean()
        if std is None:
            std = col.std()
        col = (col - mean) / std
        # add normalized columns back to dataframe
        df.loc[df[label].str.isdigit().notnull(), label] = col
        # fill missing values with 0
        df[label] = df[label].fillna(0)
    else: 
        # map columns to integers and non-valid labels to nan
        codes, uniques = pd.factorize(df[label], sort=True)
        if label in ['GENDER', 'LEFT_RIGHT']:
            assert len(uniques) == 2, f'expected 2 categories for {label}, got {len(uniques)} categories: {uniques}'
        codes = codes.astype(np.float32)
        codes[codes==-1.] = np.nan
        col = codes[~np.isnan(codes)]
        # normalize columns
        if mean is None:
            mean = col.mean()
        if std is None:
            std = col.std()
        col = (col - mean) / std
        # add normalized columns back to dataframe
        codes[~np.isnan(codes)] = col
        df[label] = codes
        # fill missing values with 0
        df[label] = df[label].fillna(0)

    return df, mean, std


def get_multi_cohort_df(data_config: Path, cohorts: Iterable[str], target_labels: Iterable[str], categories: Iterable[str], norm: str = 'macenko', feats: str = 'ctranspath', aug: str = None, clini_info: dict = {}):
    df_list = []
    np_list = []
    
    with open(data_config, 'r') as f:
        data_config = yaml.safe_load(f)
        
        for cohort in cohorts:
            clini_table = Path(data_config[cohort]['clini_table'])
            slide_csv = Path(data_config[cohort]['slide_csv'])
            feature_dir = Path(data_config[cohort]['feature_dir'][norm][feats])

            current_df = get_cohort_df(clini_table, slide_csv, feature_dir, target_labels, categories, cohort, clini_info) 
            df_list.append(current_df)
            np_list.append(len(current_df.PATIENT))

    data = pd.concat(df_list, ignore_index=True)
    
    if clini_info:
        for label in clini_info.keys():
            data, mean, std = transform_clini_info(data, label, clini_info[label]['mean'], clini_info[label]['std'])
            clini_info[label]['mean'] = mean
            clini_info[label]['std'] = std

    if len(data.PATIENT) != sum(np_list):
        print(f'number of patients in joint dataset {len(data.PATIENT)} is not equal to the sum of each dataset {sum(np_list)}')
    
    return data, clini_info


class MILDatasetIndices(Dataset):
    def __init__(self, data: pd.DataFrame, indices: Iterable[int], target_labels: Iterable[str], clini_info: dict = {}, num_tiles: int = -1, pad_tiles: bool=True, norm: str = 'macenko'):
        self.data = data.iloc[indices]
        self.indices = indices
        self.target_labels = target_labels
        self.clini_info = clini_info
        self.norm = norm

        self.num_tiles = num_tiles
        self.pad_tiles = pad_tiles

    def __getitem__(self, item):
        # load features and coords from .h5 file
        h5_path = self.data.slide_path[self.indices[item]][0]
        h5_file = h5py.File(h5_path)
        if (self.norm == 'histaugan' or self.norm == 'efficient_histaugan') and torch.rand((1,)) < 0.5:
            if 'feats_aug' in h5_file.keys():
                features = torch.Tensor(np.array(h5_file['feats_aug']))
            elif 'histaugan' in h5_file.keys():
                features = torch.Tensor(np.array(h5_file['histaugan']))
            else:
                features = torch.Tensor(np.array(h5_file['augmented']))
            version = torch.randint(features.shape[0], (1,)).item()
            features = features[version]
        else: 
            features = torch.Tensor(np.array(h5_file['feats']))
        if 'coords' in h5_file.keys():
            coords = torch.Tensor(np.array(h5_file['coords']))
        else:
            coords = 0  # NoneType is not accepted by dataloader
            
        # avoid CUDA OOM
        if features.shape[0] > 14000:
            feat_idxs = torch.randperm(features.shape[0])[:14000]
            features = features[feat_idxs]

        # randomly sample num_tiles tiles, if #tiles < num_tiles, fill vector with 0s 
        tiles = torch.tensor([features.shape[0]])
        if self.num_tiles > 0:
            if features.shape[0] <= self.num_tiles and self.pad_tiles:
                pad = torch.zeros((self.num_tiles, features.shape[1]))
                pad[:features.shape[0]] = features
                features = pad
                # also pad the coords vector, for stacking in dataloader
                pad_coords = torch.zeros((self.num_tiles, 2))
                pad_coords[:coords.shape[0]] = coords
                coords = pad_coords
            else: 
                feat_idxs = torch.randperm(features.shape[0])[:self.num_tiles]
                features = features[feat_idxs]
                coords = coords[feat_idxs]
        
        label_dict = {
            'Not mut.': 0,
            'Mutat.': 1,
            'nonMSIH': 0,
            'MSIH': 1,
            'WT': 0,
            'MUT': 1,
            'wt': 0,
            'MT': 1,
            'left': 1,
            'right': 0,
            'female': 1,
            'male': 0,
        }

        # create binary or numeric labels from categorical labels
        label = [label_dict[self.data[target][self.indices[item]]] for target in self.target_labels]
        label = torch.Tensor(label)  # .squeeze(0)

        # add clinical information to feature vector
        if self.clini_info:
            for info in self.clini_info.keys():
                clini_info = torch.Tensor([self.data[info][self.indices[item]]]).unsqueeze(0).repeat_interleave(features.shape[0], dim=0)
                features = torch.concat((features, clini_info), dim=1)

        patient = self.data.PATIENT[self.indices[item]]

        return features, coords, label, tiles, patient

    def __len__(self):
        return len(self.data)


class MILDataset(Dataset):
    def __init__(self, data_config: Path, cohorts: Iterable[str], target_labels: Iterable[str], categories: Iterable[str], norm: str = 'macenko', feats: str = 'retccl',
                 clini_info: dict = {}, num_tiles: int=-1):
        self.cohorts = cohorts
        self.clini_info = clini_info
        self.norm = norm

        df_list = []
        np_list = []
        with open(data_config, 'r') as f:
            data_config = yaml.safe_load(f)
            
            for cohort in cohorts:
                clini_table = Path(data_config[cohort]['clini_table'])
                slide_csv = Path(data_config[cohort]['slide_csv'])
                feature_dir = Path(data_config[cohort]['feature_dir'][norm][feats]) 

                current_df = self.get_cohort_df(clini_table, slide_csv, feature_dir, target_labels, categories, cohort, clini_info)
                df_list.append(current_df)
                np_list.append(len(current_df.PATIENT))

        self.data = pd.concat(df_list, ignore_index=True)
        if len(self.data.PATIENT) != sum(np_list):
            print(f'number of patients in joint dataset {len(self.data.PATIENT)} is not equal to the sum of each dataset {sum(np_list)}')
        
        if self.clini_info:
            for label in self.clini_info.keys():
                self.data, mean, std = transform_clini_info(self.data, label, self.clini_info[label]['mean'], self.clini_info[label]['std'])
                self.clini_info[label]['mean'] = mean
                self.clini_info[label]['std'] = std

        self.target_labels = [target_labels] if type(target_labels) is str else target_labels
        self.num_tiles = num_tiles

    def __getitem__(self, item):
        # load features and coords from .h5 file
        h5_path = self.data.slide_path[item][0]
        h5_file = h5py.File(h5_path)
        if self.norm == 'histaugan' and torch.rand((1,)) < 0.5:
            domain = torch.randint(7, (1,))
            if 'feats_aug' in h5_file.keys():
                features = torch.Tensor(np.array(h5_file['feats_aug']))[domain]
            else:
                features = torch.Tensor(np.array(h5_file['augmented']))[domain]
        else: 
            features = torch.Tensor(np.array(h5_file['feats']))
        if len(features.shape) == 3:
            features = features.squeeze(0)
            
        # avoid CUDA OOM
        if features.shape[0] > 14000:
            feat_idxs = torch.randperm(features.shape[0])[:14000]
            features = features[feat_idxs]
        
        if 'coords' in h5_file.keys():
            coords = torch.Tensor(np.array(h5_file['coords']))
        else:
            coords = 0  # NoneType is not accepted by dataloader

        # randomly sample num_tiles tiles, if #tiles < num_tiles, fill vector with 0s 
        tiles = torch.tensor([features.shape[0]])
        if self.num_tiles > 0:
            if features.shape[0] <= self.num_tiles:
                pad = torch.zeros((self.num_tiles, features.shape[1]))
                pad[:features.shape[0]] = features
                features = pad
                # also pad the coords vector, for stacking in dataloader
                pad_coords = torch.zeros((self.num_tiles, 2))
                pad_coords[:coords.shape[0]] = coords
                coords = pad_coords
            else: 
                feat_idxs = torch.randperm(features.shape[0])[:self.num_tiles]
                features = features[feat_idxs]
                coords = coords[feat_idxs]
        
        label_dict = {
            'Not mut.': 0,
            'Mutat.': 1,
            'nonMSIH': 0,
            'MSIH': 1,
            'WT': 0,
            'MUT': 1,
            'wt': 0,
            'MT': 1,
            'left': 1,
            'right': 0,
            'female': 1,
            'male': 0,
        }

        # create binary or numeric labels from categorical labels
        label = [label_dict[self.data[target][item]] for target in self.target_labels]
        label = torch.Tensor(label)  # .squeeze(0)

        # add clinical information to feature vector
        if self.clini_info:
            for info in self.clini_info.keys():
                clini_info = torch.Tensor([self.data[info][item]]).unsqueeze(0).repeat_interleave(features.shape[0], dim=0)
                features = torch.concat((features, clini_info), dim=1)

        patient = self.data.PATIENT[item]

        return features, coords, label, tiles, patient

    def __len__(self):
        return len(self.data)

    def get_cohort_df(self,
                      clini_table: Path, slide_csv: Path, feature_dir: Path,
                      target_labels: Iterable[str], categories: Iterable[str], cohort: str, clini_info: dict = {}) -> pd.DataFrame:
        clini_df = pd.read_csv(clini_table, dtype=str) if Path(clini_table).suffix == '.csv' else pd.read_excel(
            clini_table, dtype=str)
        slide_df = pd.read_csv(slide_csv, dtype=str)
        df = clini_df.merge(slide_df, on='PATIENT')
        # adapt dataframe to case sensitive clini tables
        # TODO implement case insensitive keys
        df = df.rename({
            'MSI': 'isMSIH',
            'Sex': 'GENDER',
            'Age': 'AGE',
            # 'Tumor Site'
        }, axis=1)

        # remove columns not in target_labels
        for key in df.columns:
            if key not in target_labels + ['PATIENT', 'SLIDE', 'FILENAME', *list(clini_info.keys())]:
                df.drop(key, axis=1, inplace=True)
        # remove rows/slides with non-valid labels
        for target in target_labels:
            df = df[df[target].isin(categories)]
        # remove slides we don't have
        h5s = set(feature_dir.glob('**/*.h5'))
        assert h5s, f'no features found in {feature_dir}!'
        h5_df = pd.DataFrame(h5s, columns=['slide_path'])
        # h5_df['FILENAME'] = h5_df.slide_path.map(lambda p: p.stem.split('.')[0])
        h5_df['FILENAME'] = h5_df.slide_path.map(lambda p: p.stem.split('.')[0]) if cohort=='TCGA' else h5_df.slide_path.map(lambda p: p.stem)
        df = df.merge(h5_df, on='FILENAME')

        # reduce to one row per patient with list of slides in `df['slide_path']`
        patient_df = df.groupby('PATIENT').first().drop(columns='slide_path')
        patient_slides = df.groupby('PATIENT').slide_path.apply(list)
        df = patient_df.merge(patient_slides, left_on='PATIENT', right_index=True).reset_index()

        return df
