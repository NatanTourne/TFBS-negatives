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

# Hardcoded arguments
datafolder = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr"
TF = "ZBTB33"
celltype = "A549"
neg_mode = "neighbors"
devices = 1
cross_val_set = 4
learning_rate = 0.0001
n_blocks = 2
target_hsize = 128
batch_size = 256
output_dir = "/data/home/natant/Negatives/Runs/Prelim_run_1_DEBUG"
early_stop_patience = 20
early_stop_metric = "AUROC"
early_stop_mode = "max"
group_name = "prelim_run_1_debug"

date = datetime.now().strftime("%Y%m%d_%H:%M")

file = f"{datafolder}/{celltype}.h5t"
Dmod = DataModule(file, TF=TF, batch_size=batch_size, neg_mode=neg_mode, cross_val_set=cross_val_set)
model = TFmodel(
    learning_rate=learning_rate,
    n_blocks=n_blocks,
    target_hsize=target_hsize
)

checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=output_dir,
        filename=celltype + "_" + TF + "_" + neg_mode + "_CV-" + str(cross_val_set) + "_" + date + '_{epoch:02d}_{val_loss:.2f}_{AUROC:.2f}',
        mode="min"
        )

early_stop = EarlyStopping(early_stop_metric, patience=early_stop_patience, mode=early_stop_mode)

callback_list = [checkpoint_callback, early_stop]


trainer = pl.Trainer(
    max_steps=5_000_000,
    accelerator="gpu",
    devices=devices,
    callbacks=callback_list,
    max_epochs=3
)

trainer.fit(model, Dmod)


do_test = True
if do_test:
    best_model_path = checkpoint_callback.best_model_path  # Re-fetch after training
    if best_model_path:
        print(f"Loading best model from: {best_model_path}")
        best_model = TFmodel.load_from_checkpoint(best_model_path)
        trainer.test(best_model, datamodule=Dmod)
    else:
        warnings.warn("No best model found — skipping test.")