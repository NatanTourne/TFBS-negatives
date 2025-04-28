import h5torch
import numpy as np
import torch
import pytorch_lightning as pl
import warnings
from pytorch_lightning.callbacks import Callback
from TFBS_negatives.data import DataModule_sanity_check
import pytorch_lightning as pl
from TFBS_negatives.models import multilabel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument("--TF", type=str, required=True, help="Transcription factor (TF) to use.")
    parser.add_argument("--celltype", type=str, required=True, help="Cell type (name before .h5t in the file).")
    parser.add_argument("--neg_mode", type=str, required=True, help="Negative sampling mode.")
    parser.add_argument("--devices", type=int, nargs='+', required=True, help="List of GPU devices to use.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--output_dir", type=str, default="/data/home/natant/Negatives/testing_ground/junk", help="Directory to save model checkpoints.")
    parser.add_argument("--early_stop_patience", type=int, default=20, help="Patience for early stopping.")
    parser.add_argument("--early_stop_metric", type=str, default="AUROC", help="Metric to monitor for early stopping.")
    parser.add_argument("--early_stop_mode", type=str, default="max", help="Mode for early stopping (min or max).")

    args = parser.parse_args()

    date = datetime.now().strftime("%Y%m%d_%H:%M")

    file = f"/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5/{args.celltype}.h5t"
    Dmod = DataModule_sanity_check(file, TF=args.TF, batch_size=args.batch_size, neg_mode=args.neg_mode)
    model = multilabel(latent_vector_size=1)

    checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath="args.output_dir,",
            filename=args.celltype + "_" + args.TF + "_" + date + 'val_loss-{epoch:02d}-{val_loss:.2f}-{AUROC: .2f}',
            mode="min"
            )

    early_stop = EarlyStopping(args.early_stop_matric, patience=args.early_stop_patience, mode=args.early_stop_mode)

    callback_list = [checkpoint_callback, early_stop]
    wandb_logger = WandbLogger(project="Negatives", entity="ntourne")

    trainer = pl.Trainer(
        max_steps=5_000_000,
        accelerator="gpu",
        devices=args.devices,
        logger=wandb_logger,
        callbacks=callback_list
    )

    trainer.fit(model, Dmod)