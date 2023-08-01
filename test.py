import argparse
import os
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from classifier import ClassifierLightning
from data import MILDataset, get_multi_cohort_df
from options import Options
from utils import save_results


def main(cfg):
    cfg.seed = torch.randint(0, 1000, (1, )).item() if cfg.seed is None else cfg.seed
    pl.seed_everything(cfg.seed, workers=True)

    # saving locations
    base_path = Path(cfg.save_dir)  # adapt to own target path
    cfg.logging_name = f'{cfg.name}_{cfg.model}_{"-".join(cfg.cohorts)}_{cfg.norm}_{cfg.target}' if not cfg.debug else 'debug'
    base_path = base_path / cfg.logging_name
    base_path.mkdir(parents=True, exist_ok=True)
    model_path = base_path / 'models'
    fold_path = base_path / 'folds'
    fold_path.mkdir(parents=True, exist_ok=True)
    result_path = base_path / 'results'
    result_path.mkdir(parents=True, exist_ok=True)

    norm_val = 'raw' if cfg.norm in ['histaugan', 'efficient_histaugan'] else cfg.norm
    norm_test = 'raw' if cfg.norm in ['histaugan', 'efficient_histaugan'] else cfg.norm

    # --------------------------------------------------------
    # load data
    # --------------------------------------------------------
    print('\n--- load dataset ---')
    categories = ['Not mut.', 'Mutat.', 'nonMSIH', 'MSIH', 'WT', 'MUT', 'wt', 'MT']
    data, clini_info = get_multi_cohort_df(
        cfg.data_config, cfg.cohorts, [cfg.target], cfg.label_dict, norm=cfg.norm, feats=cfg.feats, clini_info=cfg.clini_info
    )
    cfg.clini_info = clini_info
    cfg.input_dim += len(cfg.clini_info.keys())

    for cohort in cfg.cohorts:
        if cohort in cfg.ext_cohorts:
            cfg.ext_cohorts.pop(cfg.ext_cohorts.index(cohort))

    train_cohorts = f'{", ".join(cfg.cohorts)}'
    test_cohorts = [*cfg.ext_cohorts]
    results = {t: [] for t in test_cohorts}

    test_ext_dataloader = []
    for ext in cfg.ext_cohorts:
        test_data, clini_info = get_multi_cohort_df(
            cfg.data_config, [ext], [cfg.target], cfg.label_dict, norm=norm_test, feats=cfg.feats, clini_info=cfg.clini_info
        )
        dataset_ext = MILDataset(
            test_data,
            list(range(len(data))), [cfg.target],
            categories,
            norm=norm_test,
            feats=cfg.feats,
            clini_info=cfg.clini_info
        )
        test_ext_dataloader.append(DataLoader(dataset=dataset_ext, batch_size=1, shuffle=False, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), pin_memory=True))
    
    print(f'training cohorts: {train_cohorts}')
    print(f'testing cohorts:  {cfg.ext_cohorts}')
        
    # --------------------------------------------------------
    # k-fold cross validation
    # --------------------------------------------------------
    # skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    # patient_df = data.groupby('PATIENT').first().reset_index()
    # target_stratisfy = cfg.target if type(cfg.target) is str else cfg.target[0]
    # splits = skf.split(patient_df, patient_df[target_stratisfy])

    for l in range(cfg.folds):
        # train_idxs, val_idxs = train_test_split( train_val_idxs, stratify=patient_df.iloc[train_val_idxs][target_stratisfy], random_state=cfg.seed)
        
        # TODO implement in-domain testing, i.e. reading from csv splits

        # --------------------------------------------------------
        # model
        # --------------------------------------------------------
        cfg.pos_weight = torch.ones((1,))
        model = ClassifierLightning(cfg)

        # --------------------------------------------------------
        # model saving
        # --------------------------------------------------------
        checkpoint_callback = ModelCheckpoint(
            monitor='auroc/val' if cfg.stop_criterion == 'auroc' else 'loss/val',
            dirpath=model_path,
            filename=f'best_model_{cfg.logging_name}_fold{l}',
            save_top_k=1,
            mode='max' if cfg.stop_criterion == 'auroc' else 'min',
        )

        # --------------------------------------------------------
        # testing
        # --------------------------------------------------------
        
        trainer = pl.Trainer(
            accelerator='cpu' if socket.gethostname() == 'hpc-submit03gui' else 'auto',
            callbacks=[checkpoint_callback],
            max_epochs=cfg.num_epochs,
            val_check_interval=500,
            check_val_every_n_epoch=None,
            # limit_val_batches=0.1,  # debug
            # limit_train_batches=6,  # debug
            # limit_val_batches=6,    # debug
            # log_every_n_steps=1,  # debug
            # fast_dev_run=True,    # debug
            # max_steps=6,          # debug
            enable_model_summary=False,  # debug
        )
        
        checkpoint_path = cfg.resume if cfg.resume else model_path / f'best_model_{cfg.logging_name}_fold{l}.ckpt'

        # TODO write test function in separate file (read test split from csv files created above, load correct models for folds)
        # TODO rewrite for multiple dataloader (problem: how to save results to csv with correct name?)
        test_cohorts_dataloader = [*test_ext_dataloader]
        for idx in range(len(test_cohorts)):
            results_test = trainer.test(
                model,
                test_cohorts_dataloader[idx],
                ckpt_path=checkpoint_path,
            )
            results[test_cohorts[idx]].append(results_test[0])
            # save patient predictions to outputs csv file
            model.outputs.to_csv(result_path / f'fold{l}' / f'outputs_{test_cohorts[idx]}.csv')
                        
    # save results to csv file
    save_results(cfg, results, base_path, train_cohorts, test_cohorts)

if __name__ == '__main__':
    parser = Options()
    args = parser.parse()
    
    # Load the configuration from the YAML file
    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Update the configuration with the values from the argument parser
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name != 'config_file':
            config[arg_name]['value'] = getattr(args, arg_name)
    
    # Create a flat config file without descriptions
    config = {k: v['value'] for k, v in config.items()}

    print('\n--- load options ---')
    for name, value in sorted(config.items()):
        print(f'{name}: {str(value)}')
    
    config = argparse.Namespace(**config)
    main(config)

