import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from TFBS_negatives.data import DataModule
from TFBS_negatives.models import TFmodel
import wandb

celltype = "K562"
TF = "CTCF"

sweep_config = {
    'method': 'grid',  # or 'random', 'grid'
    'metric': {
        'name': 'val_loss',  # change this if you're logging a different metric
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-3, 1e-4, 1e-5, 1e-6]
        },
        'target_hsize': {
            'values': [32, 64, 128]
        },
        'n_blocks': {
            'values': [1, 2, 3]
        }
    }
}


def train_sweep():
    wandb.init()
    config = wandb.config
    file = f"/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5/{celltype}.h5t"
    Dmod = DataModule(file, TF=TF, batch_size=256, neg_mode="shuffled", cross_val_set=0)
    model = TFmodel(
        learning_rate=config.learning_rate,
        target_hsize=config.target_hsize,
        n_blocks=config.n_blocks,
    )

    logger = WandbLogger(project="Negatives", log_model=True)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=500,
        devices=[0]
    )

    trainer.fit(model, Dmod)


sweep_id = wandb.sweep(sweep_config, project="Negatives", entity="ntourne")
wandb.agent(sweep_id, function=train_sweep, count=20)  # run 20 sweep trials
