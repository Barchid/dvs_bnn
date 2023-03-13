from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from project.datamodules.dvsdatamodule import DVSDataModule
from project.datamodules.ncaltech101_localization import NCALTECH101Localization
from project.bnn_module import BNNModule
from project.utils.transform import DVSTransform
from torch.utils.data import random_split, DataLoader, Subset

data_dir = "~/dvs_ssl/data"
transf = ["crop", "background_activity", "flip_polarity", "event_drop_2"]
learning_rate = 1e-2
epochs = 500


def main(model="18loc"):
    # seeds the random from numpy, pytorch, etc for reproductibility
    pl.seed_everything(1234)
    
    dataset = NCALTECH101Localization(
        data_dir,
        transform=DVSTransform(
            None,
            timesteps=12,
            concat_time_channels=True,
            transforms_list=["background_noise", "flip_polarity", "event_drop_2"],
        )
    )
    
    full_length = len(dataset)
    train_len = int(0.9 * full_length)
    val_len = full_length - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(
            train_set,
            batch_size=64,
            num_workers=9,
            shuffle=True,
        )
    val_loader = DataLoader(
            val_set,
            batch_size=64,
            num_workers=5,
            shuffle=False,
        )
    
    
    module = BNNModule(learning_rate=learning_rate, n_classes=datamodule.num_classes, epochs=epochs, model=model)

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
            "/datas/sandbox/sami/bnn", name=f"{name}"
        ),
        default_root_dir=f"/datas/sandbox/sami/bnn/{name}",
        precision=16,
    )

    trainer.fit(module, train_loader, val_loader)

    # report results in a txt file
    report_path = os.path.join("train_report.txt")
    report = open(report_path, "a")

    report.write(f"BNN {model} {trainer.checkpoint_callback.best_model_score}\n")
    report.flush()
    report.close()


if __name__ == "__main__":
    # main(dataset="dvsgesture")
    # main(dataset="daily_action_dvs")
    # # main(dataset="n-caltech101")
    # main(dataset="ncars")
    # main(dataset="asl-dvs")
    # main(dataset="cifar10-dvs")
    
    # main(dataset="cifar10-dvs")
    main("18loc")
    main("20loc")
