import h5torch
import numpy as np
import torch
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import warnings
from pytorch_lightning.callbacks import Callback
from TFBS_negatives.data import DataModule
import pytorch_lightning as pl
from TFBS_negatives.models import TFmodel2
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import pytorch_lightning as pl
import warnings
import numpy as np
import os
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from torchmetrics.functional.classification import multilabel_average_precision, binary_auroc, binary_average_precision, binary_accuracy, binary_matthews_corrcoef, binary_precision, binary_specificity, binary_recall
import gc
#

def get_predicts(model_ckpt_path, device=3, idx=0):
    parts = filename.split('_')

    # Extract fields from the new naming scheme:
    # CT-GM12878, TF-ELK1$(1277-1), NEG-dinucl$shuffled, CV-5, ...
    cellline = None
    tf = None
    neg_type = None
    cv_split = None

    for p in parts:
        if p.startswith('CT-'):
            cellline = p[len('CT-'):]
        elif p.startswith('TF-'):
            # restore original underscores inside TF name
            tf = p[len('TF-'):].replace('$', '_')
        elif p.startswith('NEG-'):
            # keep '-' for later convert_dict mapping, but restore '_' inside mode
            neg_type = p[len('NEG-'):].replace('$', '_')
        elif p.startswith('CV-'):
            cv_split = p[len('CV-'):]

    if neg_type == "celltype":
        cellline_file = f"/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr copy/{cellline}.h5t"
    else:
        cellline_file = f"/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr/{cellline}.h5t"


    Dmod = DataModule(cellline_file, TF=tf, batch_size=256, neg_mode=neg_type, cross_val_set=int(cv_split))
    Dmod.set_predict_set("test")
    model = TFmodel2.load_from_checkpoint(model_ckpt_path)

    trainer = pl.Trainer(
        max_steps=5_000_000,
        accelerator="gpu",
        devices=[device]
    )
    predict_outputs = trainer.predict(model, Dmod)[idx]
    return predict_outputs


cellline_file = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr/GM12878.h5t"


# Define the folder path
folder_path = "/data/home/natant/Negatives/Runs/Review_sane_model/"

# Get all files ending in .ckpt
ckpt_files = [f for f in os.listdir(folder_path) if f.endswith('.ckpt')]

# Parse each filename to extract information
data = []
negative_types = ["dinucl_sampled", "celltype", "shuffled", "dinucl_shuffled", "neighbors"]
for filename in ckpt_files:   
    parts = filename.split('_')

    # Extract fields from the new naming scheme:
    # CT-GM12878, TF-ELK1$(1277-1), NEG-dinucl$shuffled, CV-5, ...
    cellline = None
    tf = None
    neg_type = None
    cv_split = None

    for p in parts:
        if p.startswith('CT-'):
            cellline = p[len('CT-'):]
        elif p.startswith('TF-'):
            # restore original underscores inside TF name
            tf = p[len('TF-'):].replace('$', '_')
        elif p.startswith('NEG-'):
            # keep '-' for later convert_dict mapping, but restore '_' inside mode
            neg_type = p[len('NEG-'):].replace('$', '_')
        elif p.startswith('CV-'):
            cv_split = p[len('CV-'):]


    
    
    data.append({
        'filename': filename,
        'file_path': os.path.join(folder_path, filename),
        'cellline': cellline,
        'TF': tf,
        'negative_type': neg_type,
        'cv_split': cv_split
    })

# Create DataFrame
df_ckpt = pd.DataFrame(data)


output_folder = "/data/home/natant/Negatives/Runs/Review_sane_model/Results"
os.makedirs(output_folder, exist_ok=True)

cellline = 'GM12878'
neg_type_order = ['dinucl_sampled', 'neighbors', 'celltype', 'shuffled', 'dinucl_shuffled']
num_negs = len(neg_type_order)
n_bins = 40
idx = 0 # dataset

TF_list = df_ckpt['TF'].unique().tolist()

for TF in TF_list:

    fig, axes = plt.subplots(num_negs, 3, figsize=(20,22))


    for cross_val in range(5):
        selected_runs = df_ckpt[
        (df_ckpt['TF'] == TF) & 
        (df_ckpt['cellline'] == cellline) & 
        (df_ckpt['cv_split'] == str(cross_val)) & 
        (df_ckpt["negative_type"] != "HQ")
        ]
        for neg_type in neg_type_order:
            model_ckpt_path = selected_runs[selected_runs["negative_type"] == neg_type]["file_path"].values[0]
            batch_list = get_predicts(model_ckpt_path, device=3, idx=idx)
            vals = []
            targets = []
            for item in batch_list:
                if item is None:
                    continue
                logits = item["logits"].detach().cpu().numpy().reshape(-1)
                targ = item["target"].detach().cpu().numpy().reshape(-1)
                vals.append(logits)
                targets.append(targ)
            vals = np.concatenate(vals, axis=0)
            targets = np.concatenate(targets, axis=0)

            # compute sigmoided version
            probs = 1 / (1 + np.exp(-vals))

            
            axes[neg_type_order.index(neg_type),0].hist(vals, bins=n_bins)
            axes[neg_type_order.index(neg_type),0].set_title(f"raw outputs (dl {idx}) {neg_type}")
            axes[neg_type_order.index(neg_type),1].hist(probs, bins=n_bins)
            axes[neg_type_order.index(neg_type),1].set_title(f"sigmoid(raw) (dl {idx}) {neg_type}")
            axes[neg_type_order.index(neg_type),1].set_xlim(0,1)
            prob_true, prob_pred = calibration_curve(targets, probs, n_bins=n_bins, strategy="uniform")
            axes[neg_type_order.index(neg_type),2].plot(prob_pred, prob_true, marker='o', linewidth=1, label=str(cross_val))
            #axes[neg_type_order.index(neg_type),2].plot([0,1], [0,1], linestyle='--', label='Perfectly calibrated')
            axes[neg_type_order.index(neg_type),2].set_xlabel('Mean predicted probability')
            axes[neg_type_order.index(neg_type),2].set_ylabel('Fraction of positives (empirical)')
            axes[neg_type_order.index(neg_type),2].set_title(f'Reliability diagram (calibration curve) {neg_type}')
            axes[neg_type_order.index(neg_type),2].set_xlim(0,1)
            axes[neg_type_order.index(neg_type),2].set_ylim(0,1)
            axes[neg_type_order.index(neg_type),2].legend()
            axes[neg_type_order.index(neg_type),2].grid(True)

               # Clear previous variables if they exist
            if 'vals' in locals():
                del vals
            if 'targets' in locals():
                del targets
            if 'probs' in locals():
                del probs
            if 'batch_list' in locals():
                del batch_list
            gc.collect()
            torch.cuda.empty_cache()

    for neg_type in neg_type_order:
        axes[neg_type_order.index(neg_type),2].plot([0,1], [0,1], linestyle='--', label='Perfectly calibrated')
    
    fig.suptitle(f'{cellline} - {TF} - Calibration curves across CV folds', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder,  f'{cellline}_{TF}_Std_dataset_calibration.png'), dpi=150, bbox_inches='tight')

 
