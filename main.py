from pathlib import Path
import numpy as np
import argparse
import pandas as pd
import yaml

import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch.utils.data import DataLoader
import wandb

from options import Options
from data_utils import MILDataset, MILDatasetIndices, get_multi_cohort_df
from classifier import ClassifierLightning
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
    data = get_multi_cohort_df(
        cfg.cohorts, [cfg.target], categories, norm=cfg.norm, feats=cfg.feats
    )

    test_ext_dataloader = []
    for ext in cfg.ext_cohorts:
        dataset_ext = MILDataset(
            [ext], [cfg.target],
            categories,
            norm=norm_test,
            feats=cfg.feats,
            clini_info=cfg.clini_info
        )
        test_ext_dataloader.append(DataLoader(dataset=dataset_ext, batch_size=1, shuffle=False, num_workers=14, pin_memory=True))
        
    train_cohorts = f'{", ".join(cfg.cohorts)}'
    test_cohorts = [train_cohorts, *cfg.ext_cohorts]
    results = {t: [] for t in test_cohorts}

    # --------------------------------------------------------
    # k-fold cross validation
    # --------------------------------------------------------
    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    patient_df = data.groupby('PATIENT').first().reset_index()
    target_stratisfy = cfg.target if type(cfg.target) is str else cfg.target[0]
    splits = skf.split(patient_df, patient_df[target_stratisfy])

    for l, (train_val_idxs, test_idxs) in enumerate(splits):
        train_idxs, val_idxs = train_test_split( train_val_idxs, stratify=patient_df.iloc[train_val_idxs][target_stratisfy], random_state=cfg.seed)
        
        # training dataset
        train_dataset = MILDatasetIndices(
            data,
            train_idxs, [cfg.target],
            num_tiles=cfg.num_tiles,
            pad_tiles=cfg.pad_tiles,
            norm=cfg.norm
        )
        patient_df['PATIENT'][train_idxs].to_csv(fold_path / f'folds_{cfg.logging_name}_fold{l}_train.csv')
        print(f'num training samples in fold {l}: {len(train_dataset)}')
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=cfg.bs, shuffle=True, num_workers=14, pin_memory=True
        )
        
        # validation dataset
        val_dataset = MILDatasetIndices(data, val_idxs, [cfg.target], norm=norm_val)
        patient_df['PATIENT'][val_idxs].to_csv(fold_path / f'folds_{cfg.logging_name}_fold{l}_val.csv')
        print(f'num validation samples in fold {l}: {len(val_dataset)}')
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=1, shuffle=False, num_workers=14, pin_memory=True
        )

        # test dataset (in-domain)
        test_dataset = MILDatasetIndices(data, test_idxs, [cfg.target], norm=norm_test)
        patient_df['PATIENT'][test_idxs].to_csv(fold_path / f'folds_{cfg.logging_name}_fold{l}_test.csv')
        print(f'num test samples in fold {l}: {len(test_dataset)}')
        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=1, shuffle=False, num_workers=14, pin_memory=True
        )

        # idx=2 since the ouput is feats, coords, labels
        num_pos = sum([train_dataset[i][2] for i in range(len(train_dataset))])
        cfg.pos_weight = torch.Tensor((len(train_dataset) - num_pos) / num_pos)
        cfg.criterion = "BCEWithLogitsLoss"

        # --------------------------------------------------------
        # model
        # --------------------------------------------------------
        model = ClassifierLightning(cfg)

        # --------------------------------------------------------
        # logging
        # --------------------------------------------------------
        logger = WandbLogger(
            project=cfg.project,
            name=f'{cfg.logging_name}_fold{l}',
            save_dir=cfg.save_dir,
            reinit=True,
            settings=wandb.Settings(start_method='fork'),
        )

        csv_logger = CSVLogger(
            save_dir=result_path,
            name=f'fold{l}',
        )

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
        # training
        # --------------------------------------------------------
        
        trainer = pl.Trainer(
            logger=[logger, csv_logger],
            accelerator='auto',
            precision=16,
            accumulate_grad_batches=4,
            gradient_clip_val=1,
            callbacks=[checkpoint_callback],
            max_epochs=cfg.num_epochs,
            # track_grad_norm=2,      # debug
            # num_sanity_val_steps=0,  # debug
            # val_check_interval=0.1,  # debug
            # limit_val_batches=0.1,  # debug
            # limit_train_batches=6,  # debug
            # limit_val_batches=6,    # debug
            log_every_n_steps=1,  # debug
            # fast_dev_run=True,    # debug
            # max_steps=6,          # debug
            # enable_model_summary=False,  # debug
        )

        results_val = trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
            ckpt_path=cfg.resume,
        )
        logger.log_table('results/val', results_val)
        
        # --------------------------------------------------------
        # testing
        # --------------------------------------------------------

        # TODO write test function in separate file (read test split from csv files created above, load correct models for folds)
        # TODO rewrite for multiple dataloader (problem: how to save results to csv with correct name?)
        test_cohorts_dataloader = [test_dataloader, *test_ext_dataloader]
        for idx in range(len(test_cohorts)):
            results_test = trainer.test(
                model,
                test_cohorts_dataloader[idx],
                ckpt_path='best',
            )
            results[test_cohorts[idx]].append(results_test[0])
            print(results_test[0])
            # save patient predictions to outputs csv file
            model.outputs.to_csv(result_path / f'fold{l}' / f'outputs_{test_cohorts[idx]}.csv')
            
        wandb.finish()  # required for new wandb run in next fold
            
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

