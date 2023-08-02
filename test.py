import argparse
import os
import warnings
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from classifier import ClassifierLightning
from data import MILDataset, get_multi_cohort_df
from options import Options
from utils import save_results

"""
test trained models on in-domain test set and external test set.

run file with:
python test.py --name <name-of-training-run> --config <path-to-training-config-file>
"""

# filter out UserWarnings from the torchmetrics package
warnings.filterwarnings("ignore", category=UserWarning)


def main(cfg):
    cfg.seed = torch.randint(0, 1000, (1, )).item() if cfg.seed is None else cfg.seed
    pl.seed_everything(cfg.seed, workers=True)

    base_path = Path(cfg.save_dir)
    cfg.logging_name = f'{cfg.name}_{cfg.model}_{"-".join(cfg.cohorts)}_{cfg.norm}_{cfg.target}' if cfg.name != 'debug' else 'debug'
    base_path = base_path / cfg.logging_name
    base_path.mkdir(parents=True, exist_ok=True)
    model_path = base_path / 'models'
    result_path = base_path / 'results'
    result_path.mkdir(parents=True, exist_ok=True)

    norm_test = 'raw' if cfg.norm in ['histaugan', 'efficient_histaugan'] else cfg.norm

    # --------------------------------------------------------
    # load data
    # --------------------------------------------------------

    print('\n--- load dataset ---')
    data, clini_info = get_multi_cohort_df(
        cfg.data_config,
        cfg.cohorts, [cfg.target],
        cfg.label_dict,
        norm=cfg.norm,
        feats=cfg.feats,
        clini_info=cfg.clini_info
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
        test_data, clini_info = get_multi_cohort_df(
            cfg.data_config, [ext], [cfg.target],
            cfg.label_dict,
            norm=norm_test,
            feats=cfg.feats,
            clini_info=cfg.clini_info
        )
        dataset_ext = MILDataset(
            test_data,
            list(range(len(test_data))),
            [cfg.target],
            clini_info=clini_info,
            norm=norm_test,
        )
        test_ext_dataloader.append(
            DataLoader(
                dataset=dataset_ext,
                batch_size=1,
                shuffle=False,
                num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
                pin_memory=True
            )
        )

    print(f'training cohorts: {train_cohorts}')
    print(f'testing cohorts:  {cfg.ext_cohorts}')

    # --------------------------------------------------------
    # k-fold cross validation
    # --------------------------------------------------------

    # load fold directory from data_config
    with open(cfg.data_config, 'r') as f:
        data_config = yaml.safe_load(f)
        fold_path = Path(data_config[train_cohorts]['folds']) / f"{cfg.target}_{cfg.folds}folds"
        fold_path.mkdir(parents=True, exist_ok=True)

    for k in range(cfg.folds):
        # read split from csv-file if exists already else save split to csv
        if Path(fold_path / f'fold{k}_test.csv').exists():
            test_idxs = pd.read_csv(fold_path / f'fold{k}_test.csv', index_col='Unnamed: 0').index

            # test dataset (in-domain)
            test_dataset = MILDataset(
                data, test_idxs, [cfg.target], norm=norm_test, clini_info=cfg.clini_info
            )
            print(f'num test samples in fold {k}: {len(test_dataset)}')
            test_dataloader = DataLoader(
                dataset=test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
                pin_memory=True
            )

        # --------------------------------------------------------
        # model
        # --------------------------------------------------------
        model = ClassifierLightning(cfg)

        # --------------------------------------------------------
        # testing
        # --------------------------------------------------------

        trainer = pl.Trainer(
            accelerator='auto',
            devices=1,
            max_epochs=cfg.num_epochs,
            val_check_interval=cfg.val_check_interval,
            check_val_every_n_epoch=None,
            enable_model_summary=False,
        )

        checkpoint_path = model_path / f'best_model_{cfg.logging_name}_fold{k}.ckpt'
        assert checkpoint_path.exists(), f'best model file {checkpoint_path} does not exist'

        test_cohorts_dataloader = [test_dataloader, *test_ext_dataloader]
        for idx in range(len(test_cohorts)):
            print("Testing: ", test_cohorts[idx])
            results_test = trainer.test(
                model,
                test_cohorts_dataloader[idx],
                ckpt_path=checkpoint_path,
            )
            results[test_cohorts[idx]].append(results_test[0])
            # save patient predictions to outputs csv file
            model.outputs.to_csv(result_path / f'fold{k}' / f'outputs_{test_cohorts[idx]}.csv')

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
            config[arg_name] = getattr(args, arg_name)

    print('\n--- load options ---')
    for name, value in sorted(config.items()):
        print(f'{name}: {str(value)}')

    config = argparse.Namespace(**config)
    main(config)
