from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from project.datamodules.dvsdatamodule import DVSDataModule
import tonic
from project.bnn_module import BNNModule
from project.utils.transform_robu import DVSTransformRobu

data_dir = "/home/sbarchid/dvs_ssl/data"
exp_dir = "/datas/sandbox/sami/bnn"
transf = ["crop", "background_activity", "flip_polarity", "event_drop_2"]
learning_rate = 1e-2
epochs = 500


def main(dataset, ckpt, tran, sev):
    # seeds the random from numpy, pytorch, etc for reproductibility
    pl.seed_everything(1234)

    datamodule = DVSDataModule(
        batch_size=64,
        dataset=dataset,
        timesteps=12,
        data_dir=data_dir,
        transf=transf,
        mode="cnn",
        robu = tran,
        sev=sev
    )
    
    # module = BNNModule(learning_rate=learning_rate, n_classes=datamodule.num_classes, epochs=epochs)
    module = BNNModule.load_from_checkpoint(ckpt)

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
            exp_dir, name=f"{name}"
        ),
        default_root_dir=f"{exp_dir}/{name}",
        precision=16,
    )

    # trainer.fit(module, datamodule=datamodule)
    trainer.validate(module, datamodule=datamodule)
    # report results in a txt file
    report_path = os.path.join("val_report.txt")
    report = open(report_path, "a")

    report.write(f"BNN {dataset} {trainer.checkpoint_callback.best_model_score}\n")
    report.flush()
    report.close()


if __name__ == "__main__":
    ckpt_dvsgesture = f"{exp_dir}/bnn_dvsgesture.ckpt"
    main(dataset="dvsgesture", tran="background_activity", sev=1, ckpt=ckpt_dvsgesture)
    main(dataset="dvsgesture", tran="background_activity", sev=2, ckpt=ckpt_dvsgesture)
    main(dataset="dvsgesture", tran="background_activity", sev=3, ckpt=ckpt_dvsgesture)
    main(dataset="dvsgesture", tran="background_activity", sev=4, ckpt=ckpt_dvsgesture)
    main(dataset="dvsgesture", tran="background_activity", sev=5, ckpt=ckpt_dvsgesture)
    
    main(dataset="dvsgesture", tran="occlusion", sev=1, ckpt=ckpt_dvsgesture)
    main(dataset="dvsgesture", tran="occlusion", sev=2, ckpt=ckpt_dvsgesture)
    main(dataset="dvsgesture", tran="occlusion", sev=3, ckpt=ckpt_dvsgesture)
    main(dataset="dvsgesture", tran="occlusion", sev=4, ckpt=ckpt_dvsgesture)
    main(dataset="dvsgesture", tran="occlusion", sev=5, ckpt=ckpt_dvsgesture)
    
    ckpt_daily_action_dvs = f"{exp_dir}/bnn_daily_action_dvs.ckpt"
    main(dataset="daily_action_dvs", tran="background_activity", sev=1, ckpt=ckpt_daily_action_dvs)
    main(dataset="daily_action_dvs", tran="background_activity", sev=2, ckpt=ckpt_daily_action_dvs)
    main(dataset="daily_action_dvs", tran="background_activity", sev=3, ckpt=ckpt_daily_action_dvs)
    main(dataset="daily_action_dvs", tran="background_activity", sev=4, ckpt=ckpt_daily_action_dvs)
    main(dataset="daily_action_dvs", tran="background_activity", sev=5, ckpt=ckpt_daily_action_dvs)
    
    main(dataset="daily_action_dvs", tran="occlusion", sev=1, ckpt=ckpt_daily_action_dvs)
    main(dataset="daily_action_dvs", tran="occlusion", sev=2, ckpt=ckpt_daily_action_dvs)
    main(dataset="daily_action_dvs", tran="occlusion", sev=3, ckpt=ckpt_daily_action_dvs)
    main(dataset="daily_action_dvs", tran="occlusion", sev=4, ckpt=ckpt_daily_action_dvs)
    main(dataset="daily_action_dvs", tran="occlusion", sev=5, ckpt=ckpt_daily_action_dvs)
    
    ckpt_ncaltech101 = f"{exp_dir}/bnn_ncaltech101.ckpt"
    main(dataset="n-caltech101", tran="background_activity", sev=1, ckpt=ckpt_ncaltech101)
    main(dataset="n-caltech101", tran="background_activity", sev=2, ckpt=ckpt_ncaltech101)
    main(dataset="n-caltech101", tran="background_activity", sev=3, ckpt=ckpt_ncaltech101)
    main(dataset="n-caltech101", tran="background_activity", sev=4, ckpt=ckpt_ncaltech101)
    main(dataset="n-caltech101", tran="background_activity", sev=5, ckpt=ckpt_ncaltech101)
    
    main(dataset="n-caltech101", tran="occlusion", sev=1, ckpt=ckpt_ncaltech101)
    main(dataset="n-caltech101", tran="occlusion", sev=2, ckpt=ckpt_ncaltech101)
    main(dataset="n-caltech101", tran="occlusion", sev=3, ckpt=ckpt_ncaltech101)
    main(dataset="n-caltech101", tran="occlusion", sev=4, ckpt=ckpt_ncaltech101)
    main(dataset="n-caltech101", tran="occlusion", sev=5, ckpt=ckpt_ncaltech101)
    
    ckpt_ncars = f"{exp_dir}/bnn_ncars.ckpt"
    main(dataset="ncars", tran="background_activity", sev=1, ckpt=ckpt_ncars)
    main(dataset="ncars", tran="background_activity", sev=2, ckpt=ckpt_ncars)
    main(dataset="ncars", tran="background_activity", sev=3, ckpt=ckpt_ncars)
    main(dataset="ncars", tran="background_activity", sev=4, ckpt=ckpt_ncars)
    main(dataset="ncars", tran="background_activity", sev=5, ckpt=ckpt_ncars)
    
    main(dataset="ncars", tran="occlusion", sev=1, ckpt=ckpt_ncars)
    main(dataset="ncars", tran="occlusion", sev=2, ckpt=ckpt_ncars)
    main(dataset="ncars", tran="occlusion", sev=3, ckpt=ckpt_ncars)
    main(dataset="ncars", tran="occlusion", sev=4, ckpt=ckpt_ncars)
    main(dataset="ncars", tran="occlusion", sev=5, ckpt=ckpt_ncars)
    
    # main(dataset="daily_action_dvs")
    # main(dataset="n-caltech101")
    # main(dataset="ncars")
    # main(dataset="asl-dvs")
    # main(dataset="cifar10-dvs")
    
    # main(dataset="cifar10-dvs")
