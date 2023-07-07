from pathlib import Path
from typing import Iterable, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset


"""
dataset class (MILDataset) and helper functions (get_cohort_df, transform_clini_info, get_multi_cohort_df)
"""


def get_cohort_df(clini_table: Path, slide_csv: Path, feature_dir: Path,
                    target_labels: Iterable[str], label_dict: dict, cohort: str, clini_info: dict = {}) -> pd.DataFrame:
    """
    Generate a cohort DataFrame based on clinical information and available slides.
    Adapted from https://github.com/KatherLab/marugoto/blob/main/marugoto/mil/data.py

    Args:
        clini_table (Path): Path to the clinical table file (CSV or Excel format).
        slide_csv (Path): Path to the slide CSV file.
        feature_dir (Path): Path to the directory containing slide feature files.
        target_labels (Iterable[str]): List of target labels.
        label_dict (dict): Dict of mappings from label names to numerical targets.
        cohort (str): The cohort name (e.g., 'TCGA').
        clini_info (dict, optional): Additional clinical information. Defaults to an empty dictionary.

    Returns:
        pd.DataFrame: The generated cohort DataFrame.

    Raises:
        AssertionError: If no slide features are found in the feature directory.
    """
    clini_df = pd.read_csv(clini_table, dtype=str) if Path(clini_table).suffix == '.csv' else pd.read_excel(
        clini_table, dtype=str)
    slide_df = pd.read_csv(slide_csv, dtype=str)
    df = clini_df.merge(slide_df, on='PATIENT')

    # remove columns not in target_labels
    for key in df.columns:
        if key not in target_labels + ['PATIENT', 'SLIDE', 'FILENAME', *list(clini_info.keys())]:
            df.drop(key, axis=1, inplace=True)
    # remove rows/slides with non-valid labels
    for target in target_labels:
        df = df.dropna(subset=target)
        df[target] = df[target].map(lambda p: int(p) if p.isdigit() else label_dict[p])
    # remove slides we don't have
    h5s = set(feature_dir.glob('**/*.h5'))
    assert h5s, f'no features found in {feature_dir}!'
    h5_df = pd.DataFrame(h5s, columns=['slide_path'])
    h5_df['FILENAME'] = h5_df.slide_path.map(lambda p: p.stem.split('.')[0].split('_files')[0])  # additional split('.')[0].split('_files') for TCGA cohorts
    df = df.merge(h5_df, on='FILENAME')
    # reduce to one row per patient with list of slides in `df['slide_path']`
    patient_df = df.groupby('PATIENT').first().drop(columns='slide_path')
    patient_slides = df.groupby('PATIENT').slide_path.apply(list)
    df = patient_df.merge(patient_slides, left_on='PATIENT', right_index=True).reset_index()
        
    return df


# def transform_clini_info(df: pd.DataFrame, label: str, mean: np.ndarray, std: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
#     """ 
#     Transform columns with categorical features to integers and normalize them with given mean and std dev, fill missing values.

#     Args:
#         df (pd.DataFrame): The DataFrame containing clinical information.
#         label (str): The label/column name to transform.
#         mean (np.ndarray): The mean value used for normalization.
#         std (np.ndarray): The standard deviation used for normalization.

#     Returns:
#         Tuple[pd.DataFrame, np.ndarray, np.ndarray]: A tuple containing the transformed DataFrame,
#         the updated mean value, and the updated standard deviation.

#     Raises:
#         AssertionError: If the number of unique categories for 'GENDER' or 'LEFT_RIGHT' labels is not 2.

#     """
#     # fill missing columns with 0
#     if label not in df.keys():
#         df[label] = 0
#         return df, mean, std

#     if label == 'AGE':
#         # only choose rows with valid labels
#         col = df.loc[df[label].str.isdigit().notnull(), label]
#         # map columns to integers
#         col = col.astype(int)
#         # normalize columns
#         if mean is None:
#             mean = col.mean()
#         if std is None:
#             std = col.std()
#         col = (col - mean) / std
#         # add normalized columns back to dataframe
#         df.loc[df[label].str.isdigit().notnull(), label] = col
#         # fill missing values with 0
#         df[label] = df[label].fillna(0)
#     else: 
#         # map columns to integers and non-valid labels to nan
#         codes, uniques = pd.factorize(df[label], sort=True)
#         if label in ['GENDER', 'LEFT_RIGHT']:
#             assert len(uniques) == 2, f'expected 2 categories for {label}, got {len(uniques)} categories: {uniques}'
#         codes = codes.astype(np.float32)
#         codes[codes==-1.] = np.nan
#         col = codes[~np.isnan(codes)]
#         # normalize columns
#         if mean is None:
#             mean = col.mean()
#         if std is None:
#             std = col.std()
#         col = (col - mean) / std
#         # add normalized columns back to dataframe
#         codes[~np.isnan(codes)] = col
#         df[label] = codes
#         # fill missing values with 0
#         df[label] = df[label].fillna(0)

#     return df, mean, std

def transform_clini_info(df: pd.DataFrame, clini_info: dict) -> pd.DataFrame:
    """
    Transform columns with categorical features to integers and normalize them with given mean and std dev
    """
    # take only clinical labels that are available
    columns = df.columns.tolist()
    for info in columns:
        if info in cfg.clini_info:
            # only choose rows with valid labels
            col = df.loc[df[info].str.isdigit().notnull(), info]
            # map columns to integers
            col = col.astype(int)
            # normalize columns
            mean = col.mean()
            std = col.std()
            col = (col - mean) / std
            # Adjust mean and std to desired values and update clini info dictionary
            if clini_info[info]['mean'] is None: 
                clini_info[info]['mean'] = mean
                clini_info[info]['std'] = std
                desired_mean, desired_std = 0, 1
            else: 
                desired_mean, desired_std = cfg.clini_info[info]['mean'], cfg.clini_info[info]['std']
            col = col * desired_std + desired_mean  
            # add normalized columns back to dataframe
            df.loc[df[info].str.isdigit().notnull(), info] = col


def get_multi_cohort_df(data_config: Path, cohorts: Iterable[str], target_labels: Iterable[str], label_dict: dict, norm: str = 'macenko', feats: str = 'ctranspath', clini_info: dict = {}) -> Tuple[pd.DataFrame, dict]:
    """
    Generate a multi-cohort DataFrame concatenating the DataFrame from single cohorts.

    Args:
        data_config (Path): Path to the data configuration file.
        cohorts (Iterable[str]): List of cohorts to include in the multi-cohort DataFrame.
        target_labels (Iterable[str]): List of target labels.
        label_dict (dict): Dict of mappings from label names to numerical targets.
        norm (str, optional): Normalization method. Defaults to 'macenko'.
        feats (str, optional): Feature extractor used. Defaults to 'ctranspath'.
        clini_info (dict, optional): Additional clinical information used. Defaults to an empty dictionary.

    Returns:
        Tuple[pd.DataFrame, dict]: A tuple containing the multi-cohort DataFrame and the updated clinical information.

    Raises:
        AssertionError: If the number of patients in the joint dataset does not match the sum of each individual dataset.
    """
    df_list = []
    np_list = []
    
    with open(data_config, 'r') as f:
        data_config = yaml.safe_load(f)
        
        for cohort in cohorts:
            clini_table = Path(data_config[cohort]['clini_table'])
            slide_csv = Path(data_config[cohort]['slide_csv'])
            feature_dir = Path(data_config[cohort]['feature_dir'][norm][feats])

            current_df = get_cohort_df(clini_table, slide_csv, feature_dir, target_labels, label_dict, cohort, clini_info) 
            df_list.append(current_df)
            np_list.append(len(current_df.PATIENT))

    data = pd.concat(df_list, ignore_index=True)
    
    if clini_info:
        data, mean, std = transform_clini_info(data, cfg, clini_info[label]['mean'], clini_info[label]['std'])

    if len(data.PATIENT) != sum(np_list):
        print(f'number of patients in joint dataset {len(data.PATIENT)} is not equal to the sum of each dataset {sum(np_list)}')
    
    return data, clini_info


class MILDataset(Dataset):
    """
    Dataset class for working with MIL (Multiple Instance Learning) datasets, i.e., prediction of targets for sequences of feature vectors.

    Args:
        data (pd.DataFrame): The input data containing the labels and paths to feature vectors.
        indices (Iterable[int]): The indices to select from the input data.
        target_labels (Iterable[str]): The target labels to be predicted.
        clini_info (dict, optional): Clinical information dictionary. Defaults to an empty dictionary.
        num_tiles (int, optional): The number of tiles to sample. Defaults to -1 (all tiles).
        pad_tiles (bool, optional): Whether to pad tiles with zeros if the number of tiles is less than `num_tiles`.
            Defaults to True.
        norm (str, optional): The normalization method for features. Defaults to 'macenko'.

    Methods:
        __getitem__(self, item):
            Retrieves a specific item from the dataset.

        __len__(self):
            Returns the length of the dataset.

    """

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
        # load augmented features if augmentation is used
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

        # randomly sample num_tiles tiles, if #tiles < num_tiles, fill vector with 0s 
        tiles = torch.tensor([features.shape[0]])  # only needed for AttentionMIL implementation
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

        # create binary or numeric labels from categorical labels
        label = [self.data[target][self.indices[item]] for target in self.target_labels]
        label = torch.Tensor(label).long().squeeze()

        # add clinical information to feature vector
        if self.clini_info:
            for info in self.clini_info.keys():
                clini_info = torch.Tensor([self.data[info][self.indices[item]]]).unsqueeze(0).repeat_interleave(features.shape[0], dim=0)
                features = torch.concat((features, clini_info), dim=1)

        patient = self.data.PATIENT[self.indices[item]]

        return features, coords, label, tiles, patient

    def __len__(self):
        return len(self.data)
