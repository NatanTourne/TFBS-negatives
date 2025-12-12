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
from TFBS_negatives.models import TFmodel
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import pytorch_lightning as pl
import warnings
import numpy as np
import os
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from torchmetrics.functional.classification import multilabel_average_precision, binary_auroc, binary_average_precision, binary_accuracy, binary_matthews_corrcoef, binary_precision, binary_specificity, binary_recall
import gc
from tqdm import tqdm
#


def get_predicts(model_ckpt_path, predict_set, device=3, idx=False):
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
    Dmod.set_predict_set(predict_set)
    model = TFmodel.load_from_checkpoint(model_ckpt_path)

    trainer = pl.Trainer(
        max_steps=5_000_000,
        accelerator="gpu",
        devices=[device]
    )
    if idx is False:
        predict_outputs = trainer.predict(model, Dmod)
    else:
        predict_outputs = trainer.predict(model, Dmod)[idx]
    return predict_outputs


# Define the cell line file path
cellline_file = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr/GM12878.h5t"

# Define the folder path
folder_path = "/data/home/natant/Negatives/Runs/Review_rerun"

# Get all files ending in .ckpt
ckpt_files = [f for f in os.listdir(folder_path) if f.endswith('.ckpt')]

# convert the names (OLD NAMING SCHEME)
convert_dict = {
    "dinucl_sampled": "dinucl-sampled",
    "dinucl_shuffled": "dinucl-shuffled",
}

# Create output folder if it doesn't exist
output_folder = "/data/home/natant/Negatives/Runs/Review_calibration"
os.makedirs(output_folder, exist_ok=True)

# Define variables
cellline = 'GM12878'
neg_type_order = ['dinucl-sampled', 'neighbors', 'celltype', 'shuffled', 'dinucl-shuffled']


# Parse each filename to extract information (OLD NAMING SCHEME AGAIN)
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

num_negs = len(neg_type_order)
TF_list = df_ckpt['TF'].unique().tolist()
all_results = []

for TF in tqdm(TF_list, desc="Processing TFs", position=0, leave=True):
    for cross_val in tqdm(range(5), desc="Processing CV splits", position=1, leave=True):
        selected_runs = df_ckpt[
        (df_ckpt['TF'] == TF) & 
        (df_ckpt['cellline'] == cellline) & 
        (df_ckpt['cv_split'] == str(cross_val)) & 
        (df_ckpt["negative_type"] != "HQ")
        ]
        for neg_type in tqdm(neg_type_order, desc="Processing negative types", position=2, leave=True):
            if selected_runs[selected_runs["negative_type"] == neg_type].empty:
                print(f"\033[91mSkipping TF: {TF}, CV: {cross_val}, NEG: {neg_type} no run found.\033[0m")
                continue
            model_ckpt_path = selected_runs[selected_runs["negative_type"] == neg_type]["file_path"].values[0]
            # validation run
            batch_list = get_predicts(model_ckpt_path, device=3, predict_set="val")
            std_dataset = batch_list[0]
            HQ_dataset = batch_list[1]
            # std dataset
            y_hat_all = [] 
            y_all = []
            for item in std_dataset:
                if item is None:
                    continue
                logits = item["logits"].detach().cpu().reshape(-1)
                targ = item["target"].detach().cpu().reshape(-1)
                y_hat_all.append(logits)
                y_all.append(targ)
            y_hat_all = torch.concatenate(y_hat_all, axis=0)
            y_all = torch.concatenate(y_all, axis=0) 

            # threshold metrics
            thresholds = torch.arange(0.01, 1, 0.01, dtype=torch.float64)
            best_score = {"MCC": -1, "Precision": -1, "Specificity": -1, "Accuracy": -1, "Recall": -1} 
            best_thresholds = {"MCC": 0.5, "Precision": 0.5, "Specificity": 0.5, "Accuracy": 0.5, "Recall": 0.5}
            for t in thresholds:
                t=float(t)
                mcc = binary_matthews_corrcoef(y_hat_all, y_all, threshold=t)
                prec = binary_precision(y_hat_all, y_all, threshold=t)
                spec = binary_specificity(y_hat_all, y_all, threshold=t)
                acc = binary_accuracy(y_hat_all, y_all, threshold=t)
                rec = binary_recall(y_hat_all, y_all, threshold=t)

                if mcc > best_score["MCC"]:
                    best_score["MCC"], best_thresholds["MCC"] = mcc, t
                if prec > best_score["Precision"]:
                    best_score["Precision"], best_thresholds["Precision"] = prec, t
                if spec > best_score["Specificity"]:
                    best_score["Specificity"], best_thresholds["Specificity"] = spec, t
                if acc > best_score["Accuracy"]:
                    best_score["Accuracy"], best_thresholds["Accuracy"] = acc, t
                if rec > best_score["Recall"]:
                    best_score["Recall"], best_thresholds["Recall"] = rec, t

            if 'y_all' in locals():
                del y_all
            if 'y_hat_all' in locals():
                del y_hat_all


            # HQ dataset
            y_hat_all = [] 
            y_all = []
            for item in HQ_dataset:
                if item is None:
                    continue
                logits = item["logits"].detach().cpu().reshape(-1)
                targ = item["target"].detach().cpu().reshape(-1)
                y_hat_all.append(logits)
                y_all.append(targ)
            y_hat_all = torch.concatenate(y_hat_all, axis=0)
            y_all = torch.concatenate(y_all, axis=0)

            AUROC = binary_auroc(y_hat_all, y_all)
            Average_precision = binary_average_precision(y_hat_all, y_all)

            # threshold metrics
            thresholds = torch.arange(0.01, 1, 0.01, dtype=torch.float64)
            best_score_HQ = {"MCC": -1, "Precision": -1, "Specificity": -1, "Accuracy": -1, "Recall": -1} 
            best_thresholds_HQ = {"MCC": 0.5, "Precision": 0.5, "Specificity": 0.5, "Accuracy": 0.5, "Recall": 0.5}
            for t in thresholds:
                t=float(t)
                mcc = binary_matthews_corrcoef(y_hat_all, y_all, threshold=t)
                prec = binary_precision(y_hat_all, y_all, threshold=t)
                spec = binary_specificity(y_hat_all, y_all, threshold=t)
                acc = binary_accuracy(y_hat_all, y_all, threshold=t)
                rec = binary_recall(y_hat_all, y_all, threshold=t)

                if mcc > best_score_HQ["MCC"]:
                    best_score_HQ["MCC"], best_thresholds_HQ["MCC"] = mcc, t
                if prec > best_score_HQ["Precision"]:
                    best_score_HQ["Precision"], best_thresholds_HQ["Precision"] = prec, t
                if spec > best_score_HQ["Specificity"]:
                    best_score_HQ["Specificity"], best_thresholds_HQ["Specificity"] = spec, t
                if acc > best_score_HQ["Accuracy"]:
                    best_score_HQ["Accuracy"], best_thresholds_HQ["Accuracy"] = acc, t
                if rec > best_score_HQ["Recall"]:
                    best_score_HQ["Recall"], best_thresholds_HQ["Recall"] = rec, t

            if 'y_all' in locals():
                del y_all
            if 'y_hat_all' in locals():
                del y_hat_all
            if "batch_list" in locals():
                del batch_list
            gc.collect()
            torch.cuda.empty_cache()

            # test run
            batch_list = get_predicts(model_ckpt_path, device=3, predict_set="test")
            std_dataset = batch_list[0]
            HQ_dataset = batch_list[1]
            # std dataset
            y_hat_all = [] 
            y_all = []
            for item in std_dataset:
                if item is None:
                    continue
                logits = item["logits"].detach().cpu().reshape(-1)
                targ = item["target"].detach().cpu().reshape(-1)
                y_hat_all.append(logits)
                y_all.append(targ)
            y_hat_all = torch.concatenate(y_hat_all, axis=0)
            y_all = torch.concatenate(y_all, axis=0)

            test_scores = {}
            test_scores["AUROC"] = binary_auroc(y_hat_all, y_all)
            test_scores["AvP"] = binary_average_precision(y_hat_all, y_all)
            test_scores["MCC"] = binary_matthews_corrcoef(y_hat_all, y_all, threshold=best_thresholds["MCC"])
            test_scores["Precision"] = binary_precision(y_hat_all, y_all, threshold=best_thresholds["Precision"])
            test_scores["Specificity"] = binary_specificity(y_hat_all, y_all, threshold=best_thresholds["Specificity"])
            test_scores["Accuracy"] = binary_accuracy(y_hat_all, y_all, threshold=best_thresholds["Accuracy"])
            test_scores["Recall"] = binary_recall(y_hat_all, y_all, threshold=best_thresholds["Recall"])
            
            if 'y_all' in locals():
                del y_all
            if 'y_hat_all' in locals():
                del y_hat_all

            # HQ dataset
            y_hat_all = [] 
            y_all = []
            for item in HQ_dataset:
                if item is None:
                    continue
                logits = item["logits"].detach().cpu().reshape(-1)
                targ = item["target"].detach().cpu().reshape(-1)
                y_hat_all.append(logits)
                y_all.append(targ)
            y_hat_all = torch.concatenate(y_hat_all, axis=0)
            y_all = torch.concatenate(y_all, axis=0)
            test_scores_HQ = {}
            test_scores_HQ["AUROC"] = binary_auroc(y_hat_all, y_all)
            test_scores_HQ["AvP"] = binary_average_precision(y_hat_all, y_all)
            test_scores_HQ["MCC"] = binary_matthews_corrcoef(y_hat_all, y_all, threshold=best_thresholds["MCC"])
            test_scores_HQ["Precision"] = binary_precision(y_hat_all, y_all, threshold=best_thresholds["Precision"])
            test_scores_HQ["Specificity"] = binary_specificity(y_hat_all, y_all, threshold=best_thresholds["Specificity"])
            test_scores_HQ["Accuracy"] = binary_accuracy(y_hat_all, y_all, threshold=best_thresholds["Accuracy"])
            test_scores_HQ["Recall"] = binary_recall(y_hat_all, y_all, threshold=best_thresholds["Recall"])

            test_scores_HQ_adjusted = {}
            test_scores_HQ_adjusted["AUROC"] = binary_auroc(y_hat_all, y_all)
            test_scores_HQ_adjusted["AvP"] = binary_average_precision(y_hat_all, y_all)
            test_scores_HQ_adjusted["MCC"] = binary_matthews_corrcoef(y_hat_all, y_all, threshold=best_thresholds_HQ["MCC"])
            test_scores_HQ_adjusted["Precision"] = binary_precision(y_hat_all, y_all, threshold=best_thresholds_HQ["Precision"])
            test_scores_HQ_adjusted["Specificity"] = binary_specificity(y_hat_all, y_all, threshold=best_thresholds_HQ["Specificity"])
            test_scores_HQ_adjusted["Accuracy"] = binary_accuracy(y_hat_all, y_all, threshold=best_thresholds_HQ["Accuracy"])
            test_scores_HQ_adjusted["Recall"] = binary_recall(y_hat_all, y_all, threshold=best_thresholds_HQ["Recall"])

            if 'y_all' in locals():
                del y_all
            if 'y_hat_all' in locals():
                del y_hat_all
            if "batch_list" in locals():
                del batch_list
            gc.collect()
            torch.cuda.empty_cache()

            results_dict = {
                "TF": TF,
                "cross_val": cross_val,
                "neg_type": neg_type,
                "cell_line": cellline}

            results_dict.update({
                f"val_best_thresholds_{k}": v for k, v in best_thresholds.items()
            })
            results_dict.update({
                f"val_best_scores_{k}": v for k, v in best_score.items()
            })
            results_dict.update({
                f"val_best_thresholds_HQ_{k}": v for k, v in best_thresholds_HQ.items()
            })
            results_dict.update({
                f"val_best_scores_HQ_{k}": v for k, v in best_score_HQ.items()
            })
            results_dict.update({
                f"test_scores_{k}": v for k, v in test_scores.items()
            })
            results_dict.update({
                f"test_scores_HQ_{k}": v for k, v in test_scores_HQ.items()
            })
            results_dict.update({
                f"test_scores_HQ_adjusted_{k}": v for k, v in test_scores_HQ_adjusted.items()
            })

            all_results.append(results_dict)




results_df = pd.DataFrame(all_results)
results_df.to_pickle(os.path.join(output_folder, "calibration_results_take_2.pkl"))



# if item is None:
#                     continue
#                 logits = item["logits"].detach().cpu().numpy().reshape(-1)
#                 targ = item["target"].detach().cpu().numpy().reshape(-1)
#                 y_hat_all.append(logits)
#                 y_all.append(targ)
#             y_hat_all = np.concatenate(y_hat_all, axis=0)
#             y_all = np.concatenate(y_all, axis=0) 