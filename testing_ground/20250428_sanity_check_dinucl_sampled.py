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
date = datetime.now().strftime("%Y%m%d_%H:%M")


file = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5/A549.h5t"
Dmod = DataModule_sanity_check(file, TF = "CTCF", batch_size=256, neg_mode="dinucl_sampled")
model = multilabel(latent_vector_size=1)

checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath="/data/home/natant/Negatives/testing_ground/junk",
            filename='val_loss-'+date+'-{epoch:02d}-{val_loss:.2f}-{AUROC: .2f}',
            mode="min"
            )

early_stop = EarlyStopping("AUROC", patience=20, mode="max")

callback_list = [checkpoint_callback, early_stop]
wandb_logger = WandbLogger(project="Negatives", entity="ntourne")

trainer = pl.Trainer(
    max_steps = 5_000_000,
    accelerator = "gpu",
    devices = [6],
    logger=wandb_logger,
    callbacks=callback_list
)

trainer.fit(model, Dmod)