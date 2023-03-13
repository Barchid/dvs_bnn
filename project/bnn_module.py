from argparse import ArgumentParser
from os import times
from spikingjelly.datasets.n_mnist import NMNIST

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
import torchmetrics
from project.models.resnet18 import resnet18_encoder


class BNNModule(pl.LightningModule):
    def __init__(self, learning_rate: float, n_classes: int, epochs: int, model="18", **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        if "20" in model:
            self.encoder = None
            self.fc = nn.Linear(64, n_classes)
        else:
            self.encoder = resnet18_encoder(24)
            self.fc = nn.Linear(512, n_classes)
        self.epochs = epochs

    def forward(self, x):
        # (T, B, C, H, W) --> (B, num_classes)
        x = self.encoder(x)
        y_hat = self.fc(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.accuracy(y_hat.clone().detach(), y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=False)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.accuracy(y_hat.clone().detach(), y)

        # logs
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.accuracy(y_hat.clone().detach(), y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        return [optimizer], [scheduler]
