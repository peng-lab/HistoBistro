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

    model_path = Path(cfg.model_path)
    cfg.name = f'test_{model_path.stem}' if cfg.name is None else cfg.name
    cfg.logging_name = f'{cfg.name}_{cfg.model}_{"-".join(cfg.cohorts)}_{cfg.norm}_{cfg.target}' if cfg.name != 'debug' else 'debug'

    result_path = Path(cfg.save_dir) / cfg.logging_name
    result_path.mkdir(parents=True, exist_ok=True)

    norm_test = 'raw' if cfg.norm in ['histaugan', 'efficient_histaugan'] else cfg.norm

    # --------------------------------------------------------
    # load external test data
    # --------------------------------------------------------

    print('\n--- load dataset ---')

    test_cohorts = [*cfg.ext_cohorts]
    results = {t: [] for t in test_cohorts}

    test_dataloader = []
    for ext in cfg.ext_cohorts:
        test_data, clini_info = get_multi_cohort_df(
            cfg.data_config, [ext], [cfg.target],
            cfg.label_dict,
            norm=norm_test,
            feats=cfg.feats,
            clini_info=cfg.clini_info
        )
        test_dataset = MILDataset(
            test_data,
            list(range(len(test_data))),
            [cfg.target],
            clini_info=clini_info,
            norm=norm_test,
        )
        test_dataloader.append(
            DataLoader(
                dataset=test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
                pin_memory=True
            )
        )

    print(f'testing cohorts:  {cfg.ext_cohorts}')

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------
    cfg.pos_weight = torch.ones(1)
    model = ClassifierLightning(cfg)
    weights = torch.load(model_path)
    model.load_state_dict(weights)

    # --------------------------------------------------------
    # testing
    # --------------------------------------------------------

    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        enable_model_summary=False,
    )

    test_cohorts_dataloader = [*test_dataloader]
    for idx in range(len(test_cohorts)):
        print("Testing: ", test_cohorts[idx])
        results_test = trainer.test(
            model,
            test_cohorts_dataloader[idx],
            # ckpt_path=model_path,
        )
        results[test_cohorts[idx]].append(results_test[0])
        # save patient predictions to outputs csv file
        model.outputs.to_csv(result_path / f'outputs_{test_cohorts[idx]}.csv')

    # save results to csv file
    save_results(cfg, results, result_path, 'CRCCancerCell', test_cohorts)


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
