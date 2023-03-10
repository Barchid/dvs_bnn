from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from project.datamodules.dvsdatamodule import DVSDataModule

from project.bnn_module import BNNModule

data_dir = "/sandbox0/sami/data"
transf = ["crop", "background_activity", "flip_polarity", "event_drop_2"]
learning_rate = 1e-2
epochs = 500


def main(dataset):
    # seeds the random from numpy, pytorch, etc for reproductibility
    pl.seed_everything(1234)

    datamodule = DVSDataModule(
        batch_size=64,
        dataset=dataset,
        timesteps=12,
        data_dir=data_dir,
        transf=transf,
        mode="cnn",
    )
    
    module = BNNModule(learning_rate=learning_rate, n_classes=datamodule.num_classes, epochs=epochs)

    # saves the best model checkpoint based on the accuracy in the validation set
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        filename="{epoch:03d}_{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    name = f"{dataset}"
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[
            checkpoint_callback,
            # EarlyStopping(monitor="val_acc", mode="max", patience=50),
        ],
        logger=pl.loggers.TensorBoardLogger(
            "/sandbox0/sami/experiments/bnn", name=f"{name}"
        ),
        default_root_dir=f"/sandbox0/sami/experiments/bnn/{name}",
        precision=16,
    )

    trainer.fit(module, datamodule=datamodule)

    # report results in a txt file
    report_path = os.path.join("train_report.txt")
    report = open(report_path, "a")

    report.write(f"BNN {dataset} {trainer.checkpoint_callback.best_model_score}\n")


if __name__ == "__main__":
    main(dataset="dvsgesture")
    main(dataset="daily_action_dvs")
    main(dataset="n-caltech101")
    main(dataset="ncars")
    main(dataset="asl-dvs")
    # main(dataset="cifar10-dvs")
    
    # main(dataset="cifar10-dvs")
