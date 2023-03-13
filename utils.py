import importlib
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
#from model.base_aggregator import BaseAggregator


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


# TODO: finish get_model
def get_model(model_name):
    """
    Import the module "model/[model_name].py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of TissueDataset,
    and it is case-insensitive.
    """
    model_filename = "model.aggregator" + model_name
    model_library = importlib.import_module(model_filename)

    model = None
    target_model_name = model_name.replace('_', '') + 'Model'
    for name, cls in model_library.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseAggregator):
            model = cls

    if model is None:
        raise NotImplementedError("Model does not exist!")

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


def get_scheduler(name, optimizer, **kwargs):
    # Check if the name is a valid scheduler name
    if name in lr_scheduler.__dict__:
        # Get the scheduler class from the torch.optim.lr_scheduler module
        scheduler_class = getattr(lr_scheduler, name)
        # Instantiate the scheduler with the optimizer and other keyword arguments
        scheduler = scheduler_class(optimizer, **kwargs)
        # Return the scheduler
        return scheduler
    else:
        # Raise an exception if the name is not valid
        raise ValueError(f"Invalid scheduler name: {name}")