from pathlib import Path
import argparse
import yaml

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb

from options import Options
from data_utils import MILDataset, MILDatasetIndices, get_multi_cohort_df
from classifier import ClassifierLightning

import optuna
from optuna.integration import PyTorchLightningPruningCallback

EPOCHS = 8
CLIP = 1
MODEL_DIR = 'home/ubuntu/logs/hyperparameter_tuning'
"""
in case of the following error:
AttributeError: 'Trainer' object has no attribute 'training_type_plugin'
caused by version mismatch of optuna and pytorch_lightning
change l.99 in /home/ubuntu/Miniconda3-py39_4.12.0-Linux-x86_64/envs/denbi/lib/python3.10/site-packages/optuna/integration/pytorch_lightning.py to:
should_stop = trainer.should_stop
"""
# TODO make nice and functional, so far tune for lr and wd but without visualizing the results

parser = Options()
args = parser.parser.parse_args('')

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

cfg = argparse.Namespace(**config)

# setup
cfg.seed = torch.randint(0, 1000, (1, )).item() if cfg.seed is None else cfg.seed
pl.seed_everything(cfg.seed, workers=True)

# saving locations
base_path = Path(cfg.save_dir)  # adapt to own target path
logging_name = f'{cfg.name}_{cfg.model}_{"-".join(cfg.cohorts)}_{cfg.norm}_{cfg.target}' if not cfg.debug else 'debug'
base_path = base_path / logging_name
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
data = get_multi_cohort_df(cfg.cohorts, [cfg.target], categories, norm=cfg.norm, feats=cfg.feats)

test_ext_dataloader = []
for ext in cfg.ext_cohorts:
    dataset_ext = MILDataset(
        [ext], [cfg.target], categories, norm=norm_test, feats=cfg.feats, clini_info=cfg.clini_info
    )
    test_ext_dataloader.append(
        DataLoader(
            dataset=dataset_ext, batch_size=1, shuffle=False, num_workers=14, pin_memory=True
        )
    )

train_cohorts = f'{", ".join(cfg.cohorts)}'
test_cohorts = [train_cohorts, *cfg.ext_cohorts]
results = {t: [] for t in test_cohorts}

patient_df = data.groupby('PATIENT').first().reset_index()
target_stratisfy = cfg.target if type(cfg.target) is str else cfg.target[0]
train_idxs, val_idxs = train_test_split(
    range(len(patient_df)), stratify=patient_df[target_stratisfy], random_state=cfg.seed
)

# training dataset
train_dataset = MILDatasetIndices(
    data, train_idxs, [cfg.target], num_tiles=cfg.num_tiles, pad_tiles=cfg.pad_tiles, norm=cfg.norm
)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=cfg.bs, shuffle=True, num_workers=14, pin_memory=True
)

# validation dataset
val_dataset = MILDatasetIndices(data, val_idxs, [cfg.target], norm=norm_val)
val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=1, shuffle=False, num_workers=14, pin_memory=True
)

# idx=2 since the ouput is feats, coords, labels
num_pos = sum([train_dataset[i][2] for i in range(len(train_dataset))])
cfg.pos_weight = torch.Tensor((len(train_dataset) - num_pos) / num_pos)
cfg.criterion = "BCEWithLogitsLoss"

# logger = WandbLogger(
#     project=cfg.project,
#     name=f'hyperparams_tuning',
#     save_dir=cfg.save_dir,
#     reinit=True,
#     settings=wandb.Settings(start_method='fork'),
# )


def objective(trial: optuna.trial.Trial) -> float:
    # checkpoint_callback = ModelCheckpoint(Path(MODEL_DIR) / f"trial_{trial.number}", monitor="auroc/val")
    torch.cuda.empty_cache()

    lr = trial.suggest_float("learning_rate", 1e-6, 1e-1)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-1)

    model = ClassifierLightning(cfg)
    logger = WandbLogger(
        project=cfg.project,
        name=f'hyperparams_tuning_{trial.number}',
        save_dir=cfg.save_dir,
        reinit=True,
        settings=wandb.Settings(start_method='fork'),
        offline=True,
    )

    trainer = pl.Trainer(
        logger=logger,
        precision='16-mixed',
        accelerator='auto',
        max_epochs=EPOCHS,
        gradient_clip_val=CLIP,
        enable_model_summary=False,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="auroc/val")],
    )
    hyperparameters = dict(lr=lr, wd=wd)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, train_dataloader, val_dataloader)
    return trainer.callback_metrics["auroc/val"].detach().item()


if __name__ == "__main__":
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