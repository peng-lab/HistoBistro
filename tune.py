import os
from pathlib import Path
import argparse
import warnings
import pandas as pd
import yaml

import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb
from options import Options
from data_utils import MILDataset, get_multi_cohort_df
from classifier import ClassifierLightning

import optuna
from optuna.integration import PyTorchLightningPruningCallback

"""
run file with 

python tune.py --config_file config_staging.yaml --num_epochs 10

in case of the following error:
AttributeError: 'Trainer' object has no attribute 'training_type_plugin'
caused by version mismatch of optuna and pytorch_lightning
change l.99 in /home/ubuntu/Miniconda3-py39_4.12.0-Linux-x86_64/envs/denbi/lib/python3.10/site-packages/optuna/integration/pytorch_lightning.py to:
should_stop = trainer.should_stop
"""
# TODO make nice and functional, so far tune for lr and wd but without visualizing the results

# filter out UserWarnings from the torchmetrics package
warnings.filterwarnings("ignore", category=UserWarning)

def objective(trial: optuna.trial.Trial) -> float:
    # checkpoint_callback = ModelCheckpoint(Path(MODEL_DIR) / f"trial_{trial.number}", monitor="auroc/val")
    torch.cuda.empty_cache()
    
    # --------------------------------------------------------
    # choose params
    # --------------------------------------------------------
    val_metric = 'acc/val'
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-2)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2)
    # optimizer = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "SGD"])
    # lr_scheduler = trial.suggest_categorical("lr_scheduler", ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR"])
    heads = trial.suggest_categorical("heads", [4, 8, 12])
    dim_head = trial.suggest_categorical("dim_heads", [32, 64, 128])
    
    # --------------------------------------------------------
    # load data
    # --------------------------------------------------------
    print('\n--- load dataset ---')
    data, clini_info = get_multi_cohort_df(
        cfg.data_config, cfg.cohorts, [cfg.target], cfg.label_dict, norm=cfg.norm, feats=cfg.feats, clini_info=cfg.clini_info
    )
    cfg.clini_info = clini_info
    cfg.input_dim += len(cfg.clini_info.keys())

    train_cohorts = f'{", ".join(cfg.cohorts)}'
    # --------------------------------------------------------
    # k-fold cross validation
    # --------------------------------------------------------
    # load fold directory from data_config
    with open(cfg.data_config, 'r') as f:
        data_config = yaml.safe_load(f)
        fold_path = Path(data_config[train_cohorts]['folds']) / f"{cfg.target}_{cfg.folds}folds"
        fold_path.mkdir(parents=True, exist_ok=True)
        
    # split data stratified by the labels
    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    patient_df = data.groupby('PATIENT').first().reset_index()
    target_stratisfy = cfg.target if type(cfg.target) is str else cfg.target[0]
    splits = skf.split(patient_df, patient_df[target_stratisfy])
    splits = list(splits)

    val_metric_folds = []
    # training dataset
    for k in range(cfg.folds):
        # read split from csv-file if exists already else save split to csv
        if Path(fold_path / f'fold{k}_train.csv').exists():
            train_idxs = pd.read_csv(fold_path / f'fold{k}_train.csv', index_col='Unnamed: 0').index
            val_idxs = pd.read_csv(fold_path / f'fold{k}_val.csv', index_col='Unnamed: 0').index
        else:
            train_idxs, val_idxs = train_test_split(splits[k][0], stratify=patient_df.iloc[splits[k][0]][target_stratisfy], random_state=cfg.seed)
            patient_df['PATIENT'][train_idxs].to_csv(fold_path / f'fold{k}_train.csv')
            patient_df['PATIENT'][val_idxs].to_csv(fold_path / f'fold{k}_val.csv')
            patient_df['PATIENT'][splits[k][1]].to_csv(fold_path / f'fold{k}_test.csv')

        # training dataset
        train_dataset = MILDataset(
            data,
            train_idxs, [cfg.target],
            num_tiles=cfg.num_tiles,
            pad_tiles=cfg.pad_tiles,
            norm=cfg.norm,
            clini_info=cfg.clini_info
        )
        print(f'num training samples in fold {k}: {len(train_dataset)}')
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=cfg.bs, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), pin_memory=True
        )
        if len(train_dataset) < cfg.val_check_interval:
            cfg.val_check_interval = len(train_dataset)
        
        # validation dataset
        norm_val = 'raw' if cfg.norm in ['histaugan', 'efficient_histaugan'] else cfg.norm
        val_dataset = MILDataset(data, val_idxs, [cfg.target], num_tiles=cfg.num_tiles, pad_tiles=cfg.pad_tiles, norm=norm_val, clini_info=cfg.clini_info)
        print(f'num validation samples in fold {k}: {len(val_dataset)}')
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=1, shuffle=False, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), pin_memory=True
        )
        
        # class weighting for binary classification
        if cfg.task == 'binary':
            num_pos = sum([train_dataset[i][2] for i in range(len(train_dataset))])
            cfg.pos_weight = torch.Tensor((len(train_dataset) - num_pos) / num_pos)
            
        # --------------------------------------------------------
        # model
        # --------------------------------------------------------
        cfg.lr = lr
        cfg.wd = wd
        # cfg.optimizer = optimizer
        # cfg.lr_scheduler = lr_scheduler
        cfg.heads = heads
        cfg.dim_head = dim_head
        cfg.dim = heads * dim_head
        cfg.mlp_dim = heads * dim_head
        model = ClassifierLightning(cfg)
        
        # --------------------------------------------------------
        # logging
        # --------------------------------------------------------

        # logging
        config = dict(trial.params)
        config["trial.number"] = trial.number
        logger = WandbLogger(
            project=cfg.project,
            group=f'tune_{cfg.name}',
            name=f'{trial.number}_{k}',
            save_dir=cfg.save_dir,
            config=config,
            reinit=True,
            settings=wandb.Settings(start_method='fork'),
        )

        trainer = pl.Trainer(
            logger=logger,
            precision='16-mixed',
            accelerator='auto',
            devices=1,
            max_epochs=cfg.num_epochs,
            gradient_clip_val=1,
            enable_model_summary=False,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor=val_metric)],
        )
        
        hyperparameters = dict(
            lr=lr, 
            wd=wd, 
            # optimizer=optimizer, 
            # lr_scheduler=lr_scheduler,
            heads=heads, 
            dim_heads=dim_head
        )
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, train_dataloader, val_dataloader)
        
        val_metric_folds.append(trainer.callback_metrics[val_metric].detach().item())
        wandb.finish()  # required for new wandb run in next fold

    
    # return trainer.callback_metrics["acc/val"].detach().item()
    return sum(val_metric_folds) / len(val_metric_folds)


if __name__ == "__main__":
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

    global cfg
    cfg = argparse.Namespace(**config)
    
    # --------------------------------------------------------
    # set up paths
    # --------------------------------------------------------
    
    sampler = optuna.samplers.TPESampler(multivariate=True)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=50, timeout=None, gc_after_trial=True)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    wandb.init(project=cfg.project, group=f'tune_{cfg.name}', name=f'final')
    hist = optuna.visualization.plot_optimization_history(study)
    wandb.log({"optuna/plot_optimization_history": hist})
    imp = optuna.visualization.plot_param_importances(study)
    wandb.log({"optuna/plot_param_importances": imp})