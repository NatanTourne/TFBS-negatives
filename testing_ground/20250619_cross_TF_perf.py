import os
from TFBS_negatives.data import DataModule
from TFBS_negatives.models import TFmodel
import pytorch_lightning as pl
import pandas as pd
import torch
import tqdm

output_folder = "/data/home/natant/Negatives/Runs/full_run_2/cross_TF_perf"
os.makedirs(output_folder, exist_ok=True)


ckpt_files = [f for f in os.listdir('/data/home/natant/Negatives/Runs/full_run_2') if f.endswith('.ckpt')]
#print(ckpt_files)

conv_library = {
    'dinucl_sampled': 'dinucl-sampled',
    'dinucl_shuffled': 'dinucl-shuffled',
    'shuffled': 'shuffled',
    'celltype': 'celltype',
}
cell_type_to_check = 'A549'
cross_val_set = 'CV-0'


neg_mode_to_check_list = ['shuffled', 'dinucl_sampled', 'dinucl_shuffled']

for neg_mode_to_check in neg_mode_to_check_list:
    print(f"Processing negative mode: {neg_mode_to_check}")
    selected_files = []
    TF_list = []
    target_files = {}
    for file_name in ckpt_files:
        file_name_temp = file_name.split('.ckpt')[0]
        for key, value in conv_library.items():
            file_name_temp = file_name_temp.replace(key, value)
        neg_mode = file_name_temp.split('_')[-8]
        celltype = file_name_temp.split('_')[0]
        CV = file_name_temp.split('_')[-7]
        TF = '_'.join(file_name_temp.split("_")[1:-8])
        if neg_mode == conv_library[neg_mode_to_check] and celltype == cell_type_to_check and CV == cross_val_set:
            selected_files.append(file_name)
            if TF not in TF_list:
                target_files[TF] = file_name
                TF_list.append(TF)

    data_file = f"/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr/{cell_type_to_check}.h5t"
    results_dict_AUROC = {}
    results_dict_AUROC_HQ = {}

    for target_TF in tqdm(TF_list):
        file_name = target_files[target_TF]
        print(f"Processing file: {file_name}")
        file_name_temp = file_name.split('.ckpt')[0]
        for key, value in conv_library.items():
            file_name_temp = file_name_temp.replace(key, value)
        results_dict_AUROC[target_TF] = []
        results_dict_AUROC_HQ[target_TF] = []
        file = '/data/home/natant/Negatives/Runs/full_run_2/' + file_name
        
        
        for TF in TF_list:
            best_model = TFmodel.load_from_checkpoint(file)
            trainer = pl.Trainer(
                    accelerator="gpu",
                    devices=[1]
                )
            Dmod = DataModule(data_file, TF=TF, batch_size=256, neg_mode=neg_mode_to_check, cross_val_set=int(cross_val_set.split('-')[1]))
            test_out = trainer.test(best_model, datamodule=Dmod)
            results_dict_AUROC[target_TF].append(test_out[0]['test_AUROC'])
            results_dict_AUROC_HQ[target_TF].append(test_out[0]['test_AUROC_HQ'])
            print(f"Model TF: {target_TF}, Test TF: {TF}, AUROC: {test_out[0]['test_AUROC']}, AUROC_HQ: {test_out[0]['test_AUROC_HQ']}")

            del best_model
            del trainer
            del Dmod
            import gc; gc.collect()
            torch.cuda.empty_cache()
    df_auroc = pd.DataFrame(results_dict_AUROC, index=TF_list)
    df_auroc_hq = pd.DataFrame(results_dict_AUROC_HQ, index=TF_list)

    df_auroc.to_csv(os.path.join(output_folder, f"{cell_type_to_check}_{neg_mode_to_check}_{cross_val_set}_AUROC.csv"))
    df_auroc_hq.to_csv(os.path.join(output_folder, f"{cell_type_to_check}_{neg_mode_to_check}_{cross_val_set}_AUROC_HQ.csv"))


#! for celltype negs
ckpt_files = [f for f in os.listdir('/data/home/natant/Negatives/Runs/full_run_2_ct') if f.endswith('.ckpt')]
print(ckpt_files)
neg_mode_to_check_list = ['celltype']

for neg_mode_to_check in neg_mode_to_check_list:
    print(f"Processing negative mode: {neg_mode_to_check}")
    selected_files = []
    TF_list = []
    target_files = {}
    for file_name in ckpt_files:
        file_name_temp = file_name.split('.ckpt')[0]
        for key, value in conv_library.items():
            file_name_temp = file_name_temp.replace(key, value)
        neg_mode = file_name_temp.split('_')[-8]
        celltype = file_name_temp.split('_')[0]
        CV = file_name_temp.split('_')[-7]
        TF = '_'.join(file_name_temp.split("_")[1:-8])
        if neg_mode == conv_library[neg_mode_to_check] and celltype == cell_type_to_check and CV == cross_val_set:
            selected_files.append(file_name)
            if TF not in TF_list:
                target_files[TF] = file_name
                TF_list.append(TF)

    data_file = f"/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr/{cell_type_to_check}.h5t"
    results_dict_AUROC = {}
    results_dict_AUROC_HQ = {}

    for target_TF in TF_list:
        file_name = target_files[target_TF]
        print(f"Processing file: {file_name}")
        file_name_temp = file_name.split('.ckpt')[0]
        for key, value in conv_library.items():
            file_name_temp = file_name_temp.replace(key, value)
        results_dict_AUROC[target_TF] = []
        results_dict_AUROC_HQ[target_TF] = []
        file = '/data/home/natant/Negatives/Runs/full_run_2_ct/' + file_name
        
        
        for TF in TF_list:
            best_model = TFmodel.load_from_checkpoint(file)
            trainer = pl.Trainer(
                    accelerator="gpu",
                    devices=[1]
                )
            Dmod = DataModule(data_file, TF=TF, batch_size=256, neg_mode=neg_mode_to_check, cross_val_set=int(cross_val_set.split('-')[1]))
            test_out = trainer.test(best_model, datamodule=Dmod)
            results_dict_AUROC[target_TF].append(test_out[0]['test_AUROC'])
            results_dict_AUROC_HQ[target_TF].append(test_out[0]['test_AUROC_HQ'])
            print(f"Model TF: {target_TF}, Test TF: {TF}, AUROC: {test_out[0]['test_AUROC']}, AUROC_HQ: {test_out[0]['test_AUROC_HQ']}")
            del best_model
            del trainer
            del Dmod
            import gc; gc.collect()
            torch.cuda.empty_cache()

    df_auroc = pd.DataFrame(results_dict_AUROC, index=TF_list)
    df_auroc_hq = pd.DataFrame(results_dict_AUROC_HQ, index=TF_list)

    df_auroc.to_csv(os.path.join(output_folder, f"{cell_type_to_check}_{neg_mode_to_check}_{cross_val_set}_AUROC.csv"))
    df_auroc_hq.to_csv(os.path.join(output_folder, f"{cell_type_to_check}_{neg_mode_to_check}_{cross_val_set}_AUROC_HQ.csv"))    