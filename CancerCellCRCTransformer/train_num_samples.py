import argparse
import os
from pathlib import Path
import warnings

import numpy as np
import random
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from classifier import ClassifierLightning
from data import MILDataset, get_multi_cohort_df
from options import Options

# filter out UserWarnings from the torchmetrics package
warnings.filterwarnings("ignore", category=UserWarning)

"""
move to main folder and run file with 

python train_num_samples.py 

to perform analysis of the performance depending on the number of training samples.
"""


def main(cfg):
    cfg.seed = torch.randint(0, 1000, (1, )).item() if cfg.seed is None else cfg.seed
    pl.seed_everything(cfg.seed, workers=True)

    # saving locations
    base_path = Path(cfg.save_dir)  
    cfg.logging_name = f'num_samples_{cfg.name}_{cfg.model}_{"-".join(cfg.cohorts)}_{cfg.norm}_{cfg.target}' if not cfg.debug else 'debug'
    base_path = base_path / cfg.logging_name
    base_path.mkdir(parents=True, exist_ok=True)
    model_path = base_path / 'models'
    fold_path = base_path / 'folds'
    fold_path.mkdir(parents=True, exist_ok=True)
    result_path = base_path / 'results'
    result_path.mkdir(parents=True, exist_ok=True)

    norm_val = 'raw' if cfg.norm in ['histaugan', 'efficient_histaugan'] else cfg.norm
    norm_test = 'raw' if cfg.norm in ['histaugan', 'efficient_histaugan'] else cfg.norm

    # cohorts and targets
    # --- for MSI cohorts
    cfg.cohorts = ['CPTAC', 'DACHS', 'DUSSEL', 'Epi700', 'ERLANGEN', 'FOXTROT', 'MCO', 'MECC', 'MUNICH', 'QUASAR', 'RAINBOW', 'TCGA', 'TRANSCOT']
    cfg.ext_cohorts = ['YCR-BCIP-resections', 'YCR-BCIP-biopsies', 'MAINZ', 'CHINA']
    # --- for BRAF / KRAS cohorts
    # cfg.cohorts = ['DACHS', 'MCO', 'QUASAR', 'RAINBOW', 'TCGA']
    # cfg.ext_cohorts = ['Epi700']
    
    data, clini_info = get_multi_cohort_df(
        cfg.data_config, cfg.cohorts, [cfg.target], cfg.label_dict, norm=cfg.norm, feats=cfg.feats, clini_info=cfg.clini_info
    )
    cfg.clini_info = clini_info
    cfg.input_dim += len(cfg.clini_info.keys())

    for cohort in cfg.cohorts:
        if cohort in cfg.ext_cohorts:
            cfg.ext_cohorts.pop(cfg.ext_cohorts.index(cohort))

    train_cohorts = f'{", ".join(cfg.cohorts)}'

    test_ext_dataloader = []
    for ext in cfg.ext_cohorts:
        test_data, clini_info = get_multi_cohort_df(
            cfg.data_config, [ext], [cfg.target], cfg.label_dict, norm=norm_test, feats=cfg.feats, clini_info=cfg.clini_info
        )
        dataset_ext = MILDataset(
            test_data,
            list(range(len(test_data))), 
            [cfg.target],
            norm=norm_test,
            clini_info=cfg.clini_info
        )
        test_ext_dataloader.append(DataLoader(dataset=dataset_ext, batch_size=1, shuffle=False, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), pin_memory=True))
    
    print(f'training cohorts: {train_cohorts}')
    print(f'testing cohorts:  {cfg.ext_cohorts}')

    print(f'num training samples in {cfg.cohorts}: {len(data)}')

    # start training
    num_samples = [64, 128, 256, 512, 1024, 2048, 4096, 8192, len(data)] if args.num_samples is None else [args.num_samples, ]

    ext_auc_dict = {key: {n: [] for n in num_samples} for key in cfg.ext_cohorts}
    for n in num_samples:
        for k in range(5):
            train_idxs = random.sample(list(range(len(data))), n)
            train_dataset = MILDataset(
                data,
                train_idxs, [cfg.target],
                num_tiles=cfg.num_tiles,
                pad_tiles=cfg.pad_tiles,
                norm=cfg.norm,
                clini_info=cfg.clini_info
            )

            train_dataloader = DataLoader(
                dataset=train_dataset, batch_size=cfg.bs, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), pin_memory=True
            )

            if cfg.model == 'AttentionMIL':
                cfg.optim = "Adam"
                cfg.lr_scheduler = "OneCycleLR"
                cfg.steps_per_epoch = len(train_dataloader)
                cfg.pct_start = 0.25

            # class weighting for binary classification
            if cfg.task == 'binary':
                num_pos = sum([train_dataset[i][2] for i in range(len(train_dataset))])
                cfg.pos_weight = torch.Tensor((len(train_dataset) - num_pos) / num_pos)

            # --------------------------------------------------------
            # model
            # --------------------------------------------------------
            model = ClassifierLightning(cfg)

            # --------------------------------------------------------
            # training
            # --------------------------------------------------------
                
            trainer = pl.Trainer(
                accelerator='auto',
                devices=1,
                max_epochs=cfg.num_epochs,
                check_val_every_n_epoch=None,
                enable_model_summary=False,
            )

            trainer.fit(
                model,
                train_dataloader,
            )

            print('--- start testing ----')

            # test model external cohort
            for idx, ext_cohort in enumerate(cfg.ext_cohorts):
                print("Testing: ", ext_cohort)
                results_test = trainer.test(
                    model,
                    test_ext_dataloader[idx],
                )
                ext_auc_dict[ext_cohort][n].append(results_test[0]['auroc/test'])
    
        # save results in data frame 
        for ext_cohort in cfg.ext_cohorts:
            results_csv = base_path / f'results_num_samples_{ext_cohort}_{cfg.model}_{cfg.feats}.csv'
            columns = ["num samples"] + [f'fold {i}' for i in range(5)]
            new_results = pd.DataFrame(data=np.array([[n, *ext_auc_dict[ext_cohort][n]]]), columns=columns)

            # append results to existing results data frame
            if results_csv.is_file():
                results_df = pd.read_csv(results_csv, dtype=str)
                if n in results_df["num samples"].unique():
                    continue
                else:
                    results_df = results_df.append(new_results)
            else:
                results_df = new_results

            results_df.to_csv(results_csv, sep=',', index=False)

    print('external', ext_auc_dict)
    print('-------------------------')
    print(f'training for {args.num_epochs} epochs')
    print(f'lists')
    for ext_cohort in cfg.ext_cohorts:
        print(ext_cohort)
        for n in num_samples:
            print(f'{n: <4}', ext_auc_dict[ext_cohort][n])
    print('---')
    for ext_cohort in cfg.ext_cohorts:
        print(ext_cohort)
        for n in num_samples:
            print(f'{n: <4} {np.mean(np.array(ext_auc_dict[ext_cohort][n])).round(4):.4f}±{np.std(np.array(ext_auc_dict[ext_cohort][n])).round(4):.4f}')
    print('-------------------------')


if __name__ == '__main__':
    parser = Options()
    args = parser.parse()
    
    # Load the configuration from the YAML file
    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Update the configuration with the values from the argument parser
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name != 'config_file':
            config[arg_name] = getattr(args, arg_name)

    print('\n--- load options ---')
    for name, value in sorted(config.items()):
        print(f'{name}: {str(value)}')
    
    config = argparse.Namespace(**config)
    main(config)
