import h5torch
import numpy as np
import torch
import pytorch_lightning as pl
import warnings
from pytorch_lightning.callbacks import Callback
from TFBS_negatives.data import DataModule
import pytorch_lightning as pl
from TFBS_negatives.models import TFmodel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test a checkpoint")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--TF", type=str, required=True, help="Transcription factor (TF) to use.")
    parser.add_argument("--celltype", type=str, required=True, help="Cell type (name before .h5t in the file).")
    parser.add_argument("--neg_mode", type=str, required=True, help="Negative sampling mode.")
    parser.add_argument("--devices", type=int, nargs='+', required=True, help="List of GPU devices to use.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--cross_val_set", type=int, default=0, help="Which of the 6 cross val combinations to take.")
    parser.add_argument("--datafolder", type=str, default="/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr/", help="Path to the data folder.")
    parser.add_argument("--group_name", type=str, default="default", help="Group name for the model.")


    args = parser.parse_args()

    date = datetime.now().strftime("%Y%m%d_%H:%M")

    file = f"{args.datafolder}/{args.celltype}.h5t"
    Dmod = DataModule(file, TF=args.TF, batch_size=args.batch_size, neg_mode=args.neg_mode, cross_val_set=args.cross_val_set)
    model = TFmodel.load_from_checkpoint(args.ckpt_path)


    run_name = f"CROSS-CELL-PERF_GM12878-{args.celltype}_{args.TF}_{args.neg_mode}_CV{args.cross_val_set}_{date}"
    wandb_logger = WandbLogger(project="Negatives_review", entity="ntourne", config=vars(args), name=run_name, group=args.group_name)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        logger=wandb_logger
    )
    trainer.test(model, Dmod)