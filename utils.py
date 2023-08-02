import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from models.aggregators.aggregator import BaseAggregator


def get_loss(name, **kwargs):
    # Check if the name is a valid loss name
    if name in nn.__dict__:
        # Get the loss class from the torch.nn module
        loss_class = getattr(nn, name)
        # Instantiate the loss with the reduction option
        loss = loss_class(**kwargs)
        # Return the loss
        return loss
    else:
        # Raise an exception if the name is not valid
        raise ValueError(f"Invalid loss name: {name}")


def get_model(model_name, **kwargs):
    """
    Import the module "model/aggregators/[model_name.lower()].py".
    In the file, the class called model_name will
    be instantiated. It has to be a subclass of BaseAggregator,
    and it is case-sensitive.
    """
    model_filename = "models.aggregators." + model_name.lower()
    model_library = importlib.import_module(model_filename)

    model_class = None
    for name, cls in model_library.__dict__.items():
        if name == model_name and issubclass(cls, BaseAggregator):
            model_class = cls

    if model_class is None:
        raise NotImplementedError("Model does not exist!")

    model = model_class(**kwargs)

    return model


def get_optimizer(name, model, lr=0.01, wd=0.1):
    # Check if the name is a valid optimizer name
    if name in optim.__dict__:
        # Get the optimizer class from the torch.optim module
        optimizer_class = getattr(optim, name)
        # Instantiate the optimizer with the model parameters and the learning rate
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=wd)
        # Return the optimizer
        return optimizer
    else:
        # Raise an exception if the name is not valid
        raise ValueError(f"Invalid optimizer name: {name}")


def get_scheduler(name, optimizer, *args, **kwargs):
    # Check if the name is a valid scheduler name
    if name in lr_scheduler.__dict__:
        # Get the scheduler class from the torch.optim.lr_scheduler module
        scheduler_class = getattr(lr_scheduler, name)
        # Instantiate the scheduler with the optimizer and other keyword arguments
        scheduler = scheduler_class(optimizer, *args, **kwargs)
        # Return the scheduler
        return scheduler
    else:
        # Raise an exception if the name is not valid
        raise ValueError(f"Invalid scheduler name: {name}")


def save_results(cfg, results, base_path, train_cohorts, test_cohorts, mode="test"):
    # save results to dataframe
    labels_per_fold = list(results[test_cohorts[0]][0].keys())
    labels_mean_std = [f'{l} {v}' for l in labels_per_fold for v in ['mean', 'std']]
    labels = [f'{l}_fold{k}' for l in labels_per_fold for k in range(len(results[test_cohorts[0]]))]
    labels = labels_mean_std + labels
    data = [[] for k in test_cohorts]

    for idx_c, c in enumerate(test_cohorts):
        # calculate mean and std over folds
        folds = []
        for l in labels_per_fold:
            fold = [results[c][k][l] for k in range(cfg.folds)]
            folds.extend(fold)
            fold = np.array(fold)
            data[idx_c].extend((fold.mean(), fold.std()))
        data[idx_c].extend(folds)
    results_df = pd.DataFrame(data, columns=labels)
    num_cols = len(results_df.columns)

    # add other information about the training to results dataframe
    results_df['Train'] = train_cohorts
    results_df['Test'] = test_cohorts
    results_df['Target'] = cfg.target
    results_df['Normalization'] = cfg.norm
    results_df['Feature Extraction'] = cfg.feats
    results_df['Algorithm'] = cfg.model
    results_df['Comments'] = f'{cfg.logging_name}, random state for splitting {cfg.seed}'
    # reorder columns and save to csv
    cols = results_df.columns.to_list()[num_cols:] + results_df.columns.to_list()[:num_cols]
    results_df = results_df[cols]
    # append to existing dataframe
    if Path(base_path / f'results_{cfg.logging_name}.csv').is_file():
        existing = pd.read_csv(base_path / f'results_{mode}_{cfg.logging_name}.csv', sep=',')
        results_df = pd.concat([existing, results_df], ignore_index=True)
    results_df.to_csv(base_path / f'results_{mode}_{cfg.logging_name}.csv', sep=',', index=False)


# test get_model function for all models
get_model('Transformer', num_classes=4, input_dim=512)
get_model('AttentionMIL', num_classes=4, input_dim=512)
# get_model('LAMIL', num_classes=4, input_dim=512)
get_model('Perceiver', num_classes=4, input_dim=512)
get_model('TransMIL', num_classes=4, input_dim=512)
