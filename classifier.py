import pandas as pd
import torch
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F

from utils import get_model, get_loss, get_optimizer, get_scheduler
from models.aggregators.transformer import Transformer


class ClassifierLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Transformer(num_classes=config.num_classes, input_dim=config.input_dim, pool='cls')
        self.criterion = get_loss(config.criterion, pos_weight=config.pos_weight)

        self.lr = config.lr
        self.wd = config.wd
        # self.num_steps = config.num_steps

        self.acc_train = torchmetrics.Accuracy(
            task=config.task, 
            num_classes=config.num_classes
        )
        self.acc_val = torchmetrics.Accuracy(
            task=config.task, 
            num_classes=config.num_classes
        )
        self.acc_test = torchmetrics.Accuracy(
            task=config.task, 
            num_classes=config.num_classes
        )

        self.auroc_val = torchmetrics.AUROC(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.auroc_test = torchmetrics.AUROC(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.f1_val = torchmetrics.F1Score(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.f1_test = torchmetrics.F1Score(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.precision_val = torchmetrics.Precision(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.precision_test = torchmetrics.Precision(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.recall_val = torchmetrics.Recall(
            task=config.task,
            num_classes=config.num_classes,
        )
        
        self.recall_test = torchmetrics.Recall(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.specificity_val = torchmetrics.Specificity(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.specificity_test = torchmetrics.Specificity(
            task=config.task,
            num_classes=config.num_classes,
        )

    def forward(self, x):
        logits = self.model(x)
        return logits

    def configure_optimizers(self):
        optimizer = get_optimizer(
            name=self.config.optimizer,
            model=self.model, 
            lr=self.lr,
            wd=self.wd
        )
        # scheduler = self.scheduler(
        #     name=self.config.scheduler,
        #     optimizer=optimizer, 
        # )
        return [optimizer] # , [scheduler]

    def training_step(self, batch, indice):
        x, coords, y, tiles, _ = batch  # x=features, y=labels
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1, keepdim=True)
        # self.lr_schedulers().step()

        self.acc_train(preds, y)
        self.log("acc/train", self.acc_train, prog_bar=True)
        self.log("loss/train", loss, prog_bar=False)

        return loss

    def validation_step(self, batch, indice):
        x, coords, y, tiles, patient = batch  # x=features, y=labels
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1, keepdim=True)

        self.acc_val(preds, y)
        self.auroc_val(preds, y)
        self.f1_val(preds, y)
        self.precision_val(preds, y)
        self.recall_val(preds, y)
        self.specificity_val(preds, y)

        self.log("acc/val", self.acc_val, prog_bar=True)
        self.log("auroc/val", self.auroc_val, prog_bar=False)
        self.log("f1/val", self.f1_val, prog_bar=True)
        self.log("loss/val", loss, prog_bar=False)
        self.log("precision/val", self.precision_val, prog_bar=False)
        self.log("recall/val", self.recall_val, prog_bar=False)
        self.log("specificity/val", self.specificity_val, prog_bar=False)
        
        outputs = {
            'patient': patient,
            'ground_truth': y,
            'predictions': preds,
            'logits': logits,
            'correct': y == preds,
        }
        
        return outputs
        
    def on_val_epoch_end(self, outputs):
        results = {
            'patient': torch.stack([x['patient'] for x in outputs]).cpu().numpy(),
            'ground_truth': torch.stack([x['ground_truth'] for x in outputs]).cpu().numpy(),
            'predictions': torch.stack([x['predictions'] for x in outputs]).cpu().numpy(),
            'logits': torch.stack([x['logits'] for x in outputs]).cpu().numpy(),
            'correct': torch.stack([x['correct'] for x in outputs]).cpu().numpy(),
        }

        return pd.DataFrame(results)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        print('hi')
        print(dataloader_idx)
        x, coords, y, tiles, patient = batch  # x=features, y=labels
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1, keepdim=True)
        
        self.acc_test(preds, y)
        self.auroc_test(preds, y)
        self.f1_test(preds, y)
        self.precision_test(preds, y)
        self.recall_test(preds, y)
        self.specificity_test(preds, y)
        
        self.log(f'acc/test_{dataloader_idx}', self.acc_test, prog_bar=True)
        self.log(f'auroc/test_{dataloader_idx}', self.auroc_test, prog_bar=False)
        self.log(f'f1/test_{dataloader_idx}', self.f1_test, prog_bar=True)
        self.log(f'loss/test_{dataloader_idx}', loss, prog_bar=False)
        self.log(f'precision/test_{dataloader_idx}', self.precision_test, prog_bar=False)
        self.log(f'recall/test_{dataloader_idx}', self.recall_test, prog_bar=False)
        self.log(f'specificity/test_{dataloader_idx}', self.specificity_test, prog_bar=False)
        
        outputs = {
            dataloader_idx: {
                'patient': patient,
                'ground_truth': y,
                'predictions': preds,
                'logits': logits,
                'correct': y == preds,
            }
        }
        
        return outputs
        
    def on_test_end(self, x, dataloader_idx=0):
        print(x)
        results = {
            'patient': torch.stack([x[dataloader_idx]['patient'] for x in outputs]).cpu().numpy(),
            'ground_truth': torch.stack([x[dataloader_idx]['ground_truth'] for x in outputs]).cpu().numpy(),
            'predictions': torch.stack([x[dataloader_idx]['predictions'] for x in outputs]).cpu().numpy(),
            'logits': torch.stack([x[dataloader_idx]['logits'] for x in outputs]).cpu().numpy(),
            'correct': torch.stack([x[dataloader_idx]['correct'] for x in outputs]).cpu().numpy(),
        }

        return pd.DataFrame(results)
        

