import argparse
import os
from pathlib import Path
import warnings

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
from data_utils import MILDataset, MILDatasetIndices, get_multi_cohort_df
from options import Options
from utils import save_results


# filter out UserWarnings from the torchmetrics package
warnings.filterwarnings("ignore", category=UserWarning)

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
        cfg.data_config, cfg.cohorts, [cfg.target], categories, norm=cfg.norm, feats=cfg.feats, clini_info=cfg.clini_info
    )
    cfg.clini_info = clini_info
    cfg.input_dim += len(cfg.clini_info.keys())

    for cohort in cfg.cohorts:
        if cohort in cfg.ext_cohorts:
            cfg.ext_cohorts.pop(cfg.ext_cohorts.index(cohort))

    train_cohorts = f'{", ".join(cfg.cohorts)}'
    test_cohorts = [train_cohorts, *cfg.ext_cohorts]
    results = {t: [] for t in test_cohorts}

    test_ext_dataloader = []
    for ext in cfg.ext_cohorts:
        dataset_ext = MILDataset(
            cfg.data_config,
            [ext], [cfg.target],
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
    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    patient_df = data.groupby('PATIENT').first().reset_index()
    target_stratisfy = cfg.target if type(cfg.target) is str else cfg.target[0]
    splits = skf.split(patient_df, patient_df[target_stratisfy])

    for l, (train_val_idxs, test_idxs) in enumerate(splits):
        # read split from csv-file if exists already
        if Path(fold_path / f'folds_{cfg.logging_name}_fold{l}_train.csv').exists():
            train_idxs = pd.read_csv(fold_path / f'folds_{cfg.logging_name}_fold{l}_train.csv', index_col='Unnamed: 0').index
            val_idxs = pd.read_csv(fold_path / f'folds_{cfg.logging_name}_fold{l}_val.csv', index_col='Unnamed: 0').index
            test_idxs = pd.read_csv(fold_path / f'folds_{cfg.logging_name}_fold{l}_test.csv', index_col='Unnamed: 0').index
        else:
            train_idxs, val_idxs = train_test_split(train_val_idxs, stratify=patient_df.iloc[train_val_idxs][target_stratisfy], random_state=cfg.seed)
            patient_df['PATIENT'][train_idxs].to_csv(fold_path / f'folds_{cfg.logging_name}_fold{l}_train.csv')
            patient_df['PATIENT'][val_idxs].to_csv(fold_path / f'folds_{cfg.logging_name}_fold{l}_val.csv')
            patient_df['PATIENT'][test_idxs].to_csv(fold_path / f'folds_{cfg.logging_name}_fold{l}_test.csv')

        # training dataset
        train_dataset = MILDatasetIndices(
            data,
            train_idxs, [cfg.target],
            num_tiles=cfg.num_tiles,
            pad_tiles=cfg.pad_tiles,
            norm=cfg.norm,
            clini_info=cfg.clini_info
        )
        print(f'num training samples in fold {l}: {len(train_dataset)}')
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=cfg.bs, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), pin_memory=True
        )
        if len(train_dataset) < cfg.val_check_interval:
            cfg.val_check_interval = len(train_dataset)
        
        # validation dataset
        val_dataset = MILDatasetIndices(data, val_idxs, [cfg.target], norm=norm_val, clini_info=cfg.clini_info)
        print(f'num validation samples in fold {l}: {len(val_dataset)}')
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=1, shuffle=False, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), pin_memory=True
        )

        # test dataset (in-domain)
        test_dataset = MILDatasetIndices(data, test_idxs, [cfg.target], norm=norm_test, clini_info=cfg.clini_info)
        print(f'num test samples in fold {l}: {len(test_dataset)}')
        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=1, shuffle=False, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), pin_memory=True
        )

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
            mode='offline'
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
        # set up trainer
        # --------------------------------------------------------
        
        trainer = pl.Trainer(
            logger=[logger, csv_logger],
            accelerator='auto',
            callbacks=[checkpoint_callback],
            max_epochs=cfg.num_epochs,
            val_check_interval=cfg.val_check_interval,
            check_val_every_n_epoch=None,
            # limit_val_batches=0.1,  # debug
            # limit_train_batches=6,  # debug
            # limit_val_batches=6,    # debug
            # log_every_n_steps=1,  # debug
            # fast_dev_run=True,    # debug
            # max_steps=6,          # debug
            enable_model_summary=False,  # debug
        )
        
        # --------------------------------------------------------
        # training
        # --------------------------------------------------------

        if Path(model_path / f'best_model_{cfg.logging_name}_fold{l}.pth').exists():
            pass
        else: 
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
        # test_cohorts_evaluated = []
        for idx in range(len(test_cohorts)):
            # skip evaluation if it was already performed
            # if Path(result_path / f'fold{l}' / f'outputs_{test_cohorts[idx]}.csv').exists():
            #     pass
            # else: 
            print("Testing: ", test_cohorts[idx])
            # test_cohorts_evaluated.append(test_cohorts[idx])
            results_test = trainer.test(
                model,
                test_cohorts_dataloader[idx],
                ckpt_path='best',
            )
            results[test_cohorts[idx]].append(results_test[0])
            # save patient predictions to outputs csv file
            model.outputs.to_csv(result_path / f'fold{l}' / f'outputs_{test_cohorts[idx]}.csv')
        
        # remove results entries for not evaluated test cohorts
        # for test_cohort in test_cohorts:
        #     if test_cohort not in test_cohorts_evaluated:
        #         results.pop(test_cohort)
            
        wandb.finish()  # required for new wandb run in next fold
        torch.cuda.empty_cache()
            
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

