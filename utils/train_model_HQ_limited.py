import h5torch
import numpy as np
import torch
import pytorch_lightning as pl
import warnings
from pytorch_lightning.callbacks import Callback
from TFBS_negatives.data import DataModule_HQ
import pytorch_lightning as pl
from TFBS_negatives.models import TFmodel_HQ
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
import argparse
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument("--TF", type=str, required=True, help="Transcription factor (TF) to use.")
    parser.add_argument("--celltype", type=str, required=True, help="Cell type (name before .h5t in the file).")
    parser.add_argument("--devices", type=int, nargs='+', required=True, help="List of GPU devices to use.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--output_dir", type=str, default="/data/home/natant/Negatives/testing_ground/junk", help="Directory to save model checkpoints.")
    parser.add_argument("--early_stop_patience", type=int, default=20, help="Patience for early stopping.")
    parser.add_argument("--early_stop_metric", type=str, default="AUROC", help="Metric to monitor for early stopping.")
    parser.add_argument("--early_stop_mode", type=str, default="max", help="Mode for early stopping (min or max).")
    parser.add_argument("--cross_val_set", type=int, default=0, help="Which of the 6 cross val combinations to take.")
    parser.add_argument("--datafolder", type=str, default="/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr/", help="Path to the data folder.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the model.")
    parser.add_argument("--n_blocks", type=int, default=2, help="Number of blocks in the model.")
    parser.add_argument("--target_hsize", type=int, default=128, help="Target hidden size for the model.")
    parser.add_argument("--group_name", type=str, default="default", help="Group name for the model.")
    parser.add_argument("--pos_limit", type=int, help="Limit for the number of positive samples.")
    parser.add_argument("--test", action="store_true", help="Run in test mode.")

    args = parser.parse_args()

    date = datetime.now().strftime("%Y%m%d_%H:%M")

    file = f"{args.datafolder}/{args.celltype}.h5t"
    Dmod = DataModule_HQ(file, TF=args.TF, batch_size=args.batch_size, cross_val_set=args.cross_val_set, max_positives=args.pos_limit)
    model = TFmodel_HQ(
        learning_rate=args.learning_rate,
        n_blocks=args.n_blocks,
        target_hsize=args.target_hsize
    )

    checkpoint_callback = ModelCheckpoint(
            monitor='AUROC',
            dirpath=args.output_dir,
            filename=args.celltype + "_" + args.TF +"_CV-" + str(args.pos_limit)+"_pos-"+ str(args.cross_val_set) + "_" + date + '_{epoch:02d}_{val_loss:.2f}_{AUROC:.2f}',
            mode="max"
            )

    early_stop = EarlyStopping(args.early_stop_metric, patience=args.early_stop_patience, mode=args.early_stop_mode)

    callback_list = [checkpoint_callback, early_stop]
    run_name = f"{args.celltype}_{args.TF}_CV{args.cross_val_set}_{date}"
    wandb_logger = WandbLogger(project="Negatives", entity="ntourne", config=vars(args), name=run_name, group=args.group_name)


    trainer = pl.Trainer(
        max_steps=5_000_000,
        accelerator="gpu",
        devices=args.devices,
        logger=wandb_logger,
        callbacks=callback_list
    )

    trainer.fit(model, Dmod)
    if args.test:
        best_model_path = checkpoint_callback.best_model_path  # Re-fetch after training
        if best_model_path:
            print(f"Loading best model from: {best_model_path}")
            best_model = TFmodel_HQ.load_from_checkpoint(best_model_path)
            trainer.test(best_model, datamodule=Dmod)
        else:
            warnings.warn("No best model found — skipping test.")
