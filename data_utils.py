import random
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PatientFeaturesCoordsDataset(Dataset):
    def __init__(self, path_label_list, transform=None, target_transform=None, get_coords=False):
        self.path_label_list = path_label_list

        self.transform = transforms.ToTensor()

        self.target_transform = target_transform

    def __len__(self):
        return len(self.path_label_list)

    def __getitem__(self, idx):
        features_coords = torch.load(self.path_label_list[idx][0])
        stack_features, stack_coords = [], []
        for i in np.arange(0, len(features_coords)):
            stack_features.append(features_coords[i][0])
            stack_coords.append(features_coords[i][1])

        # coordinate ordering: first y then x
        stack_features = np.stack(stack_features)
        stack_coords = np.stack(stack_coords)
        ind = np.lexsort((stack_coords[:, 0], stack_coords[:, 1]))
        stack_features = stack_features[ind]
        stack_coords = stack_coords[ind]

        features = torch.from_numpy(stack_features)
        coords = torch.from_numpy(stack_coords).float()
        label = torch.tensor(self.path_label_list[idx][1])

        # randomly keep 80% of tiles
        # p = 0.5
        # idx = torch.randperm(features.size(0))[:int(features.size(0)*p)]
        # features = features[idx]

        return features, coords, label


class AugPatientFeaturesCoordsDataset(Dataset):
    def __init__(self, path_label_list, transform=None, target_transform=None, get_coords=False):

        self.path_label_list = path_label_list

        self.transform = transforms.ToTensor()

        self.target_transform = target_transform

    def __len__(self):
        return len(self.path_label_list)

    def __getitem__(self, idx):

        random_value_aug = random.uniform(0, 1)
        if random_value_aug > 0.5:
            rand_dom = random.randrange(0, 6)
            path = self.path_label_list[idx][0]
            path = path.replace('/RetCCL_512px_crc_wonorm_diag_frozen/',
                                f'/RetCCL_512px_crc_histgan_diag_frozen/domain_{rand_dom}/')
            features_coords = torch.load(path)
        else:
            features_coords = torch.load(self.path_label_list[idx][0])

        stack_features, stack_coords = [], []
        for i in np.arange(0, len(features_coords)):
            stack_features.append(features_coords[i][0])
            stack_coords.append(features_coords[i][1])

        # coordinate ordering: first y then x
        stack_features = np.stack(stack_features)
        stack_coords = np.stack(stack_coords)
        ind = np.lexsort((stack_coords[:, 0], stack_coords[:, 1]))
        stack_features = stack_features[ind]
        stack_coords = stack_coords[ind]

        features = torch.from_numpy(stack_features)
        coords = torch.from_numpy(stack_coords).float()
        label = torch.tensor(self.path_label_list[idx][1])

        # randomly keep 80% of tiles
        # p = 0.5
        # idx = torch.randperm(features.size(0))[:int(features.size(0)*p)]
        # features = features[idx]

        return features, coords, label


MSI_cohorts_dresden = {
    'Rainbow': {
        'targets': ['braf', 'isMSIH', 'kras', 'nras', 'pik3ca'],
        'clini_table': Path('/mnt/Mars_03_CRC_WSIs/Jan/desktop/RAINBOW_CRC/RAINBOW-CRCREPROCESS-MM-DX_CLINI.xlsx'),
        'slide_csv': Path('/mnt/Mars_03_CRC_WSIs/Jan/desktop/RAINBOW_CRC/RAINBOW-CRCREPROCESS-MM-DX_SLIDE.csv'),
        'feature_dir': Path('/home/janniehues/Documents/CRC_Rainbow/features/Xiyue-Wang/'),
    },
    'Quasar': {
        'targets': ['BRAF', 'isMSIH', 'KRAS', 'NRAS'],
        'clini_table': Path('/home/janniehues/Documents/CRC_Quasar/QUASAR_CLINI_DATA/QUASAR-CRC-DX_CLINI.xlsx'),
        'slide_csv': Path('/home/janniehues/Documents/CRC_Quasar/QUASAR_CLINI_DATA/QUASAR-CRC-DX_SLIDE.csv'),
        'feature_dir': Path('/home/janniehues/Documents/CRC_Quasar/features/Xiyue-Wang/'),
    }
}

MSI_cohorts_munich = {
    'Rainbow': {
        'targets': ['braf', 'isMSIH', 'kras', 'nras', 'pik3ca'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/RAINBOW-CRCREPROCESS-MM-DX_CLINI.xlsx'),
        # 'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/RAINBOW-CRCREPROCESS-DX_CLINI_MERGED.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/RAINBOW-CRCREPROCESS-DX_SLIDE.csv'),
        'feature_dir': {
            'macenko': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/RAINBOW/features/Macenko/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/RAINBOW/features/Macenko/CTransPath/'),    
            },
            'raw': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/RAINBOW/features/Raw/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/RAINBOW/features/Raw/Rainbow_HistAuGAN_CTransPath/'),
            },
            'histaugan': {
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/RAINBOW/features/Raw/Rainbow_HistAuGAN_CTransPath/'),
            },
        },
    },
    'Quasar': {
        'targets': ['BRAF', 'isMSIH', 'KRAS', 'NRAS'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/QUASAR-CRC-DX_CLINI_cliniinfo.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/QUASAR-CRC-DX_SLIDE.csv'),
        'feature_dir': {
            'macenko': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/QUASAR/features/Macenko/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/QUASAR/features/Macenko/CTransPath/'),
            },
            'raw': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/QUASAR/features/Raw/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/QUASAR/features/Raw/Quasar_HistAuGAN_CTransPath/'),
            },
            'histaugan': {
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/QUASAR/features/Raw/Quasar_HistAuGAN_CTransPath/'),
            },
        },
    },
    'Yorkshire-resections': {
        'targets': ['BRAF', 'isMSIH', 'KRAS', 'NRAS'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/YORKSHIRE-RESECTIONS-DX_CLINI.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/YORKSHIRE-RESECTIONS-DX_SLIDE.csv'),
        'feature_dir': {
            'macenko': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/YORKSHIRE/features/Macenko/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/YORKSHIRE/features/Macenko/CTransPath/')
            },
            'raw': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/YORKSHIRE/features/Raw/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/YORKSHIRE/features/Raw/Yorkshire_HistAuGAN_CTransPath/'),
            },
            'histaugan': {
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/YORKSHIRE/features/Raw/Yorkshire_HistAuGAN_CTransPath/'),
            },
        },
    },
    'Yorkshire-biopsies': {
        'targets': ['BRAF', 'isMSIH', 'KRAS', 'NRAS'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/YORKSHIRE-BIOPSIESFULL-DX_CLINI.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/YORKSHIRE-BIOPSIESFULL-DX_SLIDE.csv'),
        'feature_dir': {
            'macenko': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/YORKSHIRE/features/Macenko/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/YORKSHIRE/features/Macenko/CTransPath/')
            },
            'raw': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/YORKSHIRE/features/Raw/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/YORKSHIRE/features/Raw/Yorkshire_HistAuGAN_CTransPath/'),
            },
            'histaugan': {
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/YORKSHIRE/features/Raw/Yorkshire_HistAuGAN_CTransPath/')
            },
        },
    },
    # 'TCGA': {
    #     'targets': ['BRAF', 'isMSIH', 'KRAS', 'NRAS'],
    #     'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/TCGA-CRC-DX_CLINI.xlsx'),
    #     'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/TCGA-CRC-DX_SLIDE.csv'),
    #     'feature_dir': {
    #         'macenko': {
    #             'retccl': Path('/lustre/groups/peng/datasets/histology_data/TCGA-CRC/features/Macenko/Xiyue-Wang/'),
    #             'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/TCGA-CRC/features/Macenko/CTransPath/')
    #         },
    #         'raw': {
    #             'retccl': Path('/lustre/groups/peng/datasets/histology_data/TCGA-CRC/features/Raw/Xiyue-Wang/'),
    #             # 'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/TCGA-CRC/features/Raw/CTransPath-HistAuGAN/'),
    #             'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/TCGA-CRC/features/Raw/TCGA_HistAuGAN_CTransPath_1/'),
    #         },
    #         'histaugan': {
    #             'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/TCGA-CRC/features/Raw/TCGA_HistAuGAN_CTransPath_1/'),
    #         },
    #     },
    # },
    'TCGA': {
        'targets': ['BRAF', 'isMSIH', 'KRAS', 'NRAS'],
        'clini_table': Path('/home/ubuntu/data/TCGA-CRC/TCGA-CRC-DX_CLINI.xlsx'),
        'slide_csv': Path('/home/ubuntu/data/TCGA-CRC/TCGA-CRC-DX_SLIDE.csv'),
        'feature_dir': {
            'macenko': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/TCGA-CRC/features/Macenko/Xiyue-Wang/'),
                'ctranspath': Path('/home/ubuntu/data/TCGA-CRC/features/TCGA_Macenko_CTransPath')
            },
            'raw': {
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/TCGA-CRC/features/marugoto_histaugan'),
            },
            'histaugan': {
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/TCGA-CRC/features/marugoto_histaugan'),
            },
            'efficient_histaugan': {
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/TCGA-CRC/features/marugoto_efficient_histaugan'),
            },
        },
    },
    'CPTAC': {
        'targets': ['BRAF', 'isMSIH', 'KRAS', 'NRAS'],
        # 'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/coad_cptac_2019_clinical_data.xlsx'),
        # 'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/coad_cptac_2019_slide.csv'),
        'clini_table': Path('/home/ubuntu/data/CPTAC-COAD/coad_cptac_2019_clinical_data.xlsx'),
        'slide_csv': Path('/home/ubuntu/data/CPTAC-COAD/coad_cptac_2019_slide.csv'),
        'feature_dir': {
            'macenko': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/CPTAC/features/Macenko/RetCCL'),
                # 'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/CPTAC/features/Macenko/CTransPath/')
                'ctranspath': Path('/home/ubuntu/data/CPTAC-COAD/features/CPTAC_Macenko_CTransPath')
            },
            'raw': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/CPTAC/features/Raw/RetCCL'), 
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/CPTAC/features/Raw/CPTAC_HistAuGAN_CTransPath/'),
                'ctranspath': Path('/content/drive/MyDrive/PhD/data/CPTAC_COAD/features/CPTAC_Macenko_CTransPath'),
            },
            'histaugan': {
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/CPTAC/features/Raw/CPTAC_HistAuGAN_CTransPath/'),
            },
        },
    },
    'Munich': {
        'targets': ['BRAF', 'isMSIH', 'KRAS', 'NRAS'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/MUCBERN-CRC-DX_CLINI.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/MUCBERN-CRC-DX_SLIDE.csv'),
        'feature_dir': {
            'macenko': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/MUNICH/features/Macenko/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/MUNICH/features/Macenko/CTransPath/')
                },
            'raw': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/MUNICH/features/Raw/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/MUNICH/features/Raw/MUNICH_HistAuGAN_CTransPath/'),
            },
            'histaugan': {
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/MUNICH/features/Raw/MUNICH_HistAuGAN_CTransPath/'),

            },
        },
    },
    'MECC': {
        'targets': ['BRAF', 'isMSIH', 'KRAS', 'NRAS'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/MECC-CRCBATCH1234-DX_CLINI.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/MECC-CRCBATCH1234-DX_SLIDE.csv'),
        'feature_dir': {
            'macenko': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/MECC/features/Macenko/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/MECC/features/Macenko/CTransPath/')
                },
            'raw': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/MECC/features/Raw/RetCCL/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/MECC/features/Raw/MECC_HistAuGAN_CTransPath/'),
                },
            'histaugan': {
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/MECC/features/Raw/MECC_HistAuGAN_CTransPath/'),
                },
            },
    },
    'Duessel': {
        'targets': ['BRAF', 'isMSIH', 'KRAS', 'NRAS'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/DUSSEL-CRC-DX_CLINI.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/DUSSEL-CRC-DX_SLIDE.csv'),
        'feature_dir': {
            'macenko': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/DUSSEL/features/Macenko/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/DUSSEL/features/Macenko/CTransPath/')
                },
            'raw': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/DUSSEL/features/Raw/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/DUSSEL/features/Raw/DUSSEL_HistAuGAN_CTransPath/'),
                },
            'histaugan': {
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/DUSSEL/features/Raw/DUSSEL_HistAuGAN_CTransPath/'),
                },
            },
    },
    'Dachs': {
        'targets': ['braf', 'isMSIH', 'kras'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/DACHS-CRC-DX_CLINI.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/DACHS-CRC-DX_SLIDE.csv'),
        'feature_dir': {
            'macenko': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/DACHS/features/Macenko/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/DACHS/features/Macenko/CTransPath/')
                },
            'raw': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/DACHS/features/Raw/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/DACHS/features/Raw/Dachs_HistAuGAN_CTransPath/'),
            },
            'histaugan': {
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/DACHS/features/Raw/Dachs_HistAuGAN_CTransPath/'),
            },
        },
    },
    'Belfast': {
        'targets': ['braf', 'isMSIH', 'kras'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/BELFAST-CRC-DX_CLINI.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/BELFAST-CRC-DX_SLIDE.csv'),
        'feature_dir': {
            'macenko': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/BELFAST/features/Macenko/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/BELFAST/features/Macenko/CTransPath/')
            },
            'raw': {
                'retccl': Path('/lustre/groups/peng/datasets/histology_data/BELFAST/features/Raw/Xiyue-Wang/'),
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/BELFAST/features/Raw/BELFAST_HistAuGAN_CTransPath'),
            },
            'histaugan': {
                'ctranspath': Path('/lustre/groups/peng/datasets/histology_data/BELFAST/features/Raw/BELFAST_HistAuGAN_CTransPath/'),
            },
        },
    },
    # --- STAD cohorts ----------------------------------------------
    'BERN-STAD': {
        'targets': ['isMSIH'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/BERN-STAD-Classification-DX_CLINI.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/BERN-STAD-Classification-DX_SLIDE.csv'),
        'feature_dir': {
            'macenko': Path('/lustre/groups/peng/datasets/histology_data/BERN-STAD/features/Macenko/RetCCL/')
            },
    },
    'KCCH-STAD': {
        'targets': ['isMSIH'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/KCCH-STAD-DX_Wholeslide_CLINI.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/KCCH-STAD-DX_Wholeslide_SLIDE.csv'),
        'feature_dir': {
            'macenko': Path('/lustre/groups/peng/datasets/histology_data/KCCH-STAD/features/Macenko/RetCCL/')
            },
    },
    'KIEL-STAD': {
        'targets': ['isMSIH'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/KIEL-STAD-DX_GC_Subtype_CLINI.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/KIEL-STAD-DX-WSI_pythontiles_SLIDE.csv'),
        'feature_dir': {
            'macenko': Path('/lustre/groups/peng/datasets/histology_data/KIEL-STAD/features/Macenko/RetCCL/')
            },
    },
    'LEEDS-STAD': {
        'targets': ['isMSIH'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/LEEDS-STAD-DX_GC_Subtype_CLINI.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/LEEDS-STAD-DX-WSI_pythontiles_SLIDE.csv'),
        'feature_dir': {
            'macenko': Path('/lustre/groups/peng/datasets/histology_data/LEEDS-STAD/features/Macenko/RetCCL/')
            },
    },
    'TCGA-STAD': {
        'targets': ['isMSIH'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/TCGA-STAD-DX_CLINI.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/TCGA-STAD-Classification-DX_SLIDE.csv'),
        'feature_dir': {
            'macenko': Path('/lustre/groups/peng/datasets/histology_data/TCGA-STAD/features/Macenko/RetCCL/')
            },
    },
    'TUM-STAD': {
        'targets': ['isMSIH'],
        'clini_table': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/TUM-STAD-DX_CLINI.xlsx'),
        'slide_csv': Path('/lustre/groups/peng/datasets/histology_data/clini_tables/TUM-STAD-DX_Wholeslide_SLIDE.csv'),
        'feature_dir': {
            'macenko': Path('/lustre/groups/peng/datasets/histology_data/TUM-STAD/features/Macenko/RetCCL/')
            },
    },
}


def get_cohort_df(clini_table: Path, slide_csv: Path, feature_dir: Path,
                    target_labels: Iterable[str], categories: Iterable[str], cohort: str, clini_info: bool = False) -> pd.DataFrame:
    
    clini_df = pd.read_csv(clini_table, dtype=str) if Path(clini_table).suffix == '.csv' else pd.read_excel(
        clini_table, dtype=str)
    slide_df = pd.read_csv(slide_csv, dtype=str)
    df = clini_df.merge(slide_df, on='PATIENT')
    # adapt dataframe to case sensitive clini tables
    df = df.rename({
        'MSI': 'isMSIH',
        'BRAF': 'braf', 'BRAF_mutation': 'braf', 'braf_status': 'braf', 
        'KRAS': 'kras', 'kras_status': 'kras', 'KRAS_mutation': 'kras',
        'NRAS': 'nras', 'NRAS_mutation': 'nras',  
    }, axis=1)

    # remove columns not in target_labels
    for key in df.columns:
        if key not in target_labels + ['PATIENT', 'SLIDE', 'FILENAME', 'AGE', 'GENDER', 'LEFT_RIGHT']:
            df.drop(key, axis=1, inplace=True)
    # remove rows/slides with non-valid labels
    for target in target_labels:
        df = df[df[target].isin(categories)]
    if clini_info:
        df = df[df['GENDER'].isin(['female', 'male'])]
        df = df[df['AGE'].str.isdigit()]
        df = df[df['LEFT_RIGHT'].isin(['left', 'right'])]
    # remove slides we don't have
    h5s = set(feature_dir.glob('*.h5'))
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


def get_multi_cohort_df(cohorts: Iterable[str], target_labels: Iterable[str], categories: Iterable[str], norm: str = 'macenko', feats: str = 'ctranspath', aug: str = None, clini_info: bool = False):
    df_list = []
    np_list = []
    for cohort in cohorts:
        clini_table = MSI_cohorts_munich[cohort]['clini_table']
        slide_csv = MSI_cohorts_munich[cohort]['slide_csv']
        feature_dir = MSI_cohorts_munich[cohort]['feature_dir'][norm][feats]

        current_df = get_cohort_df(clini_table, slide_csv, feature_dir, target_labels, categories, cohort, clini_info) 
        df_list.append(current_df)
        np_list.append(len(current_df.PATIENT))

    data = pd.concat(df_list, ignore_index=True)
    if len(data.PATIENT) != sum(np_list):
        print(f'number of patients in joint dataset {len(data.PATIENT)} is not equal to the sum of each dataset {sum(np_list)}')
    
    return data


class MILDatasetIndices(Dataset):
    def __init__(self, data: pd.DataFrame, indices: Iterable[int], target_labels: Iterable[str], clini_info: bool = False, num_tiles: int = -1, pad_tiles: bool=True, norm: str = 'macenko'):
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

        # add clinical information to feature vector (GENDER, AGE, LEFT_RIGHT)
        if self.clini_info:
            gender = label_dict[self.data['GENDER'][self.indices[item]]]
            age = int(self.data['AGE'][self.indices[item]])
            loc = label_dict[self.data['LEFT_RIGHT'][self.indices[item]]]
            clini_info = torch.Tensor((gender, age, loc)).unsqueeze(0).repeat_interleave(features.shape[0], dim=0)
            features = torch.concat((features, clini_info), dim=1)

        patient = self.data.PATIENT[self.indices[item]]

        return features, coords, label, tiles, patient

    def __len__(self):
        return len(self.data)


class MILDataset(Dataset):
    def __init__(self, cohorts: Iterable[str], target_labels: Iterable[str], categories: Iterable[str], norm: str = 'macenko', feats: str = 'retccl',
                 clini_info: bool = False, num_tiles: int=-1):
        self.cohorts = cohorts
        self.clini_info = clini_info
        self.norm = norm

        df_list = []
        np_list = []
        for cohort in cohorts:
            clini_table = MSI_cohorts_munich[cohort]['clini_table']
            slide_csv = MSI_cohorts_munich[cohort]['slide_csv']
            feature_dir = MSI_cohorts_munich[cohort]['feature_dir'][norm][feats]

            current_df = self.get_cohort_df(clini_table, slide_csv, feature_dir, target_labels, categories, cohort)
            df_list.append(current_df)
            np_list.append(len(current_df.PATIENT))

        self.data = pd.concat(df_list, ignore_index=True)
        if len(self.data.PATIENT) != sum(np_list):
            print(f'number of patients in joint dataset {len(self.data.PATIENT)} is not equal to the sum of each dataset {sum(np_list)}')

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

        # add clinical information to feature vector (GENDER, AGE, LEFT_RIGHT)
        if self.clini_info:
            gender = label_dict[self.data['GENDER'][item]]
            age = int(self.data['AGE'][item])
            loc = label_dict[self.data['LEFT_RIGHT'][item]]
            clini_info = torch.Tensor((gender, age, loc)).unsqueeze(0).repeat_interleave(features.shape[0], dim=0)
            features = torch.concat((features, clini_info), dim=1)

        patient = self.data.PATIENT[item]

        return features, coords, label, tiles, patient

    def __len__(self):
        return len(self.data)

    def get_cohort_df(self,
                      clini_table: Path, slide_csv: Path, feature_dir: Path,
                      target_labels: Iterable[str], categories: Iterable[str], cohort: str) -> pd.DataFrame:
        clini_df = pd.read_csv(clini_table, dtype=str) if Path(clini_table).suffix == '.csv' else pd.read_excel(
            clini_table, dtype=str)
        slide_df = pd.read_csv(slide_csv, dtype=str)
        df = clini_df.merge(slide_df, on='PATIENT')
        # adapt dataframe to case sensitive clini tables
        df = df.rename({
            'MSI': 'isMSIH',
            'BRAF': 'braf', 'BRAF_mutation': 'braf', 'braf_status': 'braf', 
            'KRAS': 'kras', 'kras_status': 'kras', 'KRAS_mutation': 'kras',
            'NRAS': 'nras', 'NRAS_mutation': 'nras',  
        }, axis=1)

        # remove columns not in target_labels
        for key in df.columns:
            if key not in target_labels + ['PATIENT', 'SLIDE', 'FILENAME', 'AGE', 'GENDER', 'LEFT_RIGHT']:
                df.drop(key, axis=1, inplace=True)
        # remove rows/slides with non-valid labels
        for target in target_labels:
            df = df[df[target].isin(categories)]
        if self.clini_info:
            df = df[df['GENDER'].isin(['female', 'male'])]
            df = df[df['AGE'].str.isdigit()]
            df = df[df['LEFT_RIGHT'].isin(['left', 'right'])]
        # remove slides we don't have
        h5s = set(feature_dir.glob('*.h5'))
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
