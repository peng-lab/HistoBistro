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
        self.criterion = get_loss(config.criterion, config.weight)

        self.lr = config.lr
        self.wd = config.wd
        self.num_steps = config.num_steps
        
        self.outputs

        # TODO: implement model loading with argument parser


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
            threshold=config.threshold,
            num_classes=config.num_classes,
            average=config.average
        )
        self.auroc_test = torchmetrics.AUROC(
            task=config.task,
            threshold=config.threshold,
            num_classes=config.num_classes,
            average=config.average
        )

        self.f1_val = torchmetrics.F1Score(
            task=config.task,
            threshold=config.threshold,
            num_classes=config.num_classes,
            average=config.average
        )
        self.f1_test = torchmetrics.F1Score(
            task=config.task,
            threshold=config.threshold,
            num_classes=config.num_classes,
            average=config.average
        )

        self.precision_val = torchmetrics.Precision(
            task=config.task,
            threshold=config.threshold,
            num_classes=config.num_classes,
            average=config.average
        )
        self.precision_test = torchmetrics.Precision(
            task=config.task,
            threshold=config.threshold,
            num_classes=config.num_classes,
            average=config.average
        )

        self.recall_val = torchmetrics.Recall(
            task=config.task,
            threshold=config.threshold,
            num_classes=config.num_classes,
            average=config.average
        )
        self.recall_test = torchmetrics.Recall(
            task=config.task,
            threshold=config.threshold,
            num_classes=config.num_classes,
            average=config.average
        )

        self.specificity_val = torchmetrics.Specificity(
            task=config.task,
            threshold=config.threshold,
            num_classes=config.num_classes,
            average=config.average
        )
        self.specificity_test = torchmetrics.Specificity(
            task=config.task,
            threshold=config.threshold,
            num_classes=config.num_classes,
            average=config.average
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
        scheduler = self.scheduler(
            name=self.config.scheduler,
            optimizer=optimizer, 
        )
        return [optimizer], [scheduler]

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.acc_train(preds, y)
        self.log("acc/train", self.acc_train, prog_bar=True)
        self.log("loss/train", loss, prog_bar=False)

        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

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
        
    def val_epoch_end(self, outputs):
        results = {
            'patient': torch.stack([x['patient'] for x in outputs]).cpu().numpy(),
            'ground_truth': torch.stack([x['ground_truth'] for x in outputs]).cpu().numpy(),
            'predictions': torch.stack([x['predictions'] for x in outputs]).cpu().numpy(),
            'logits': torch.stack([x['logits'] for x in outputs]).cpu().numpy(),
            'correct': torch.stack([x['correct'] for x in outputs]).cpu().numpy(),
        }

        return pd.DataFrame(results)

    def test_step(self, batch):
        x, y, _, patient = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.acc_test(preds, y)
        self.auroc_test(preds, y)
        self.f1_test(preds, y)
        self.precision_test(preds, y)
        self.recall_test(preds, y)
        self.specificity_test(preds, y)
        
        self.log("acc/test", self.acc_test, prog_bar=True)
        self.log("auroc/test", self.auroc_test, prog_bar=False)
        self.log("f1/test", self.f1_test, prog_bar=True)
        self.log("loss/test", loss, prog_bar=False)
        self.log("precision/test", self.precision_test, prog_bar=False)
        self.log("recall/test", self.recall_test, prog_bar=False)
        self.log("specificity/test", self.specificity_test, prog_bar=False)
        
        outputs = {
            'patient': patient,
            'ground_truth': y,
            'predictions': preds,
            'logits': logits,
            'correct': y == preds,
        }
        
        return outputs
        
    def test_epoch_end(self, outputs):
        results = {
            'patient': torch.stack([x['patient'] for x in outputs]).cpu().numpy(),
            'ground_truth': torch.stack([x['ground_truth'] for x in outputs]).cpu().numpy(),
            'predictions': torch.stack([x['predictions'] for x in outputs]).cpu().numpy(),
            'logits': torch.stack([x['logits'] for x in outputs]).cpu().numpy(),
            'correct': torch.stack([x['correct'] for x in outputs]).cpu().numpy(),
        }

        return pd.DataFrame(results)
        
