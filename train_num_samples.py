import argparse
import os
from pathlib import Path
import warnings

import numpy as np
import random
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
from data_utils import MILDatasetIndices, get_multi_cohort_df
from options import Options
from utils import save_results

# filter out UserWarnings from the torchmetrics package
warnings.filterwarnings("ignore", category=UserWarning)


def main(cfg):
    cfg.seed = torch.randint(0, 1000, (1, )).item() if cfg.seed is None else cfg.seed
    pl.seed_everything(cfg.seed, workers=True)

    # saving locations
    base_path = Path(cfg.save_dir)  # adapt to own target path
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
    cfg.cohorts = ['CPTAC', 'DACHS', 'DUSSEL', 'Epi700', 'ERLANGEN', 'FOXTROT', 'MCO', 'MECC', 'MUNICH', 'QUASAR', 'RAINBOW', 'TCGA', 'TRANSCOT']
    cfg.ext_cohorts = ['YCR-BCIP-resections', 'YCR-BCIP-biopsies', 'MAINZ', 'CHINA']
    
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
        test_data, clini_info = get_multi_cohort_df(
            cfg.data_config, ext, [cfg.target], categories, norm=norm_test, feats=cfg.feats, clini_info=cfg.clini_info
        )
        dataset_ext = MILDatasetIndices(
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

    print(f'num training samples in {cfg.cohorts}: {len(data)}')

    # start training
    num_samples = [62, 128, 256, 512, 1024, 2048, 4096, 8192, len(data)] if args.num_samples is None else [args.num_samples, ]
    # num_samples = [32, 64, 128, 256, *list(range(500, len(dataset)+1, 500))] if args.num_samples is None else [args.num_samples, ]
    # num_samples = [50, 100, 250, 8000]
    # num_samples = [50,]
    ext_auc_dict = {key: {n: [] for n in num_samples} for key in cfg.ext_cohorts}
    for n in num_samples:
        for k in range(5):
            train_idxs = random.sample(list(range(len(data))), n)
            train_dataset = MILDatasetIndices(
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

            if len(train_dataset) < cfg.val_check_interval:
                cfg.val_check_interval = len(train_dataset)

            # if args.model == 'attmil':
            #     opt = torch.optim.Adam(m.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd)
            #     lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.num_epochs, steps_per_epoch=len(train_dataloader), pct_start=0.25)
            # else:
            #     opt = torch.optim.AdamW(m.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)

            num_pos = sum([train_dataset[i][2] for i in range(len(train_dataset))])
            cfg.pos_weight = torch.Tensor((len(train_dataset) - num_pos) / num_pos)
            cfg.criterion = "BCEWithLogitsLoss"

            # --------------------------------------------------------
            # model
            # --------------------------------------------------------
            model = ClassifierLightning(cfg)

            # --------------------------------------------------------
            # model saving
            # --------------------------------------------------------

            trainer = pl.Trainer(
                # logger=[logger, csv_logger],
                accelerator='auto',
                # callbacks=[checkpoint_callback],
                max_epochs=cfg.num_epochs,
                # val_check_interval=cfg.val_check_interval,
                check_val_every_n_epoch=None,
                # limit_val_batches=0.1,  # debug
                # limit_train_batches=6,  # debug
                # limit_val_batches=6,    # debug
                # log_every_n_steps=1,  # debug
                # fast_dev_run=True,    # debug
                # max_steps=6,          # debug
                enable_model_summary=False,  # debug
            )

            results_val = trainer.fit(
                model,
                train_dataloader,
            )

            print('--- start testing ----')

            # test model external cohort
            for idx, ext_cohort in enumerate(cfg.ext_cohorts):
                print("Testing: ", test_cohorts[idx])
                # test_cohorts_evaluated.append(test_cohorts[idx])
                results_test = trainer.test(
                    model,
                    test_ext_dataloader[idx],
                    # ckpt_path='best',
                )

                ext_auc_dict[ext_cohort][n].append(results_test[0]['auroc/test'])
    
        # save results in data frame 
        for ext_cohort in cfg.ext_cohorts:
            results_csv = result_path / f'results_num_samples_{ext_cohort}_{cfg.model}_{cfg.feats}.csv'
            columns = ["num samples"] + [f'fold {i}' for i in range(5)]
            new_results = pd.DataFrame(data=np.array([[n, *ext_auc_dict[ext_cohort][n]]]), columns=columns)

            # append results to existing results data frame
            if results_csv.is_file():
                results_df = pd.read_csv(results_csv, dtype=str)
                if n in results_df["num samples"]:
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
    for ext_cohort in ext_cohorts:
        print(ext_cohort)
        for n in num_samples:
            print(f'{n: <4}', ext_auc_dict[ext_cohort][n])
    print('---')
    for ext_cohort in ext_cohorts:
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
            config[arg_name]['value'] = getattr(args, arg_name)
    
    # Create a flat config file without descriptions
    config = {k: v['value'] for k, v in config.items()}

    print('\n--- load options ---')
    for name, value in sorted(config.items()):
        print(f'{name}: {str(value)}')
    
    config = argparse.Namespace(**config)
    main(config)
