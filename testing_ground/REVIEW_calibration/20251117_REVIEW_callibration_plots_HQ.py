import numpy as np
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from TFBS_negatives.data import DataModule
import pytorch_lightning as pl
from TFBS_negatives.models import TFmodel
import pytorch_lightning as pl
import numpy as np
import os
import gc
import torch

def get_predicts(model_ckpt_path, device=3, idx=0):
    convert_dict = {
    "dinucl_sampled": "dinucl-sampled",
    "dinucl_shuffled": "dinucl-shuffled",
}
    filename = os.path.basename(model_ckpt_path)
    for old_name, new_name in convert_dict.items():
        filename = filename.replace(old_name, new_name)
    parts = filename.split('_')
    
    # Extract cell line (first part)
    cellline = parts[0]    
    cv_split = parts[-7].split('-')[1]
    neg_type = parts[-8]
    tf = '_'.join(parts[1:-8])
    Original_neg_mode = neg_type.replace('-', '_')

    if neg_type == "celltype":
        cellline_file = f"/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr copy/{cellline}.h5t"
    else:
        cellline_file = f"/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr/{cellline}.h5t"


    Dmod = DataModule(cellline_file, TF=tf, batch_size=256, neg_mode=Original_neg_mode, cross_val_set=int(cv_split))
    model = TFmodel.load_from_checkpoint(model_ckpt_path)

    trainer = pl.Trainer(
        max_steps=5_000_000,
        accelerator="gpu",
        devices=[device]
    )
    predict_outputs = trainer.predict(model, Dmod)[idx]
    return predict_outputs


cellline_file = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr/GM12878.h5t"


# Define the folder path
folder_path = "/data/home/natant/Negatives/Runs/Review_rerun"

# Get all files ending in .ckpt
ckpt_files = [f for f in os.listdir(folder_path) if f.endswith('.ckpt')]

convert_dict = {
    "dinucl_sampled": "dinucl-sampled",
    "dinucl_shuffled": "dinucl-shuffled",
}

# Parse each filename to extract information
data = []
negative_types = ["dinucl-sampled", "celltype", "shuffled", "dinucl-shuffled", "neighbors"]
for filename in ckpt_files:   
    old_filename = filename
    for old_name, new_name in convert_dict.items():
        filename = filename.replace(old_name, new_name)
    parts = filename.split('_')
    neg_type = parts[-8]
    if neg_type not in negative_types:
        neg_type = "HQ"
        tf = '_'.join(parts[1:-7])
    else:
        tf = '_'.join(parts[1:-8])
        neg_type = parts[-8]

    
    # Extract cell line (first part)
    cellline = parts[0] 
    cv_split = parts[-7].split('-')[1]


    
    
    data.append({
        'filename': filename,
        'file_path': os.path.join(folder_path, old_filename),
        'cellline': cellline,
        'TF': tf,
        'negative_type': neg_type,
        'cv_split': cv_split
    })

# Create DataFrame
df_ckpt = pd.DataFrame(data)


output_folder = "/data/home/natant/Negatives/Runs/Review_rerun/Calibration_plots/"
os.makedirs(output_folder, exist_ok=True)

cellline = 'GM12878'
neg_type_order = ['dinucl-sampled', 'neighbors', 'celltype', 'shuffled', 'dinucl-shuffled']
num_negs = len(neg_type_order)
n_bins = 50
idx = 1 # dataset

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
    plt.savefig(os.path.join(output_folder,  f'{cellline}_{TF}_HQ_dataset_calibration.png'), dpi=450, bbox_inches='tight')

 
