import pandas as pd
import subprocess
import itertools
import os
import h5torch
from queue import Queue
from threading import Thread
import csv

# Load the missing runs from the CSV file
missing_runs_path = "/data/home/natant/Negatives/Runs/full_run_3_missing/missing_runs.csv"
datafolder = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr"
output_dir = "/data/home/natant/Negatives/Runs/full_run_3_missing"
group_name = "full_run_3"
max_concurrent_models = 4

missing_runs_df = pd.read_csv(missing_runs_path)

# Create a queue for the missing runs
missing_queue = Queue()

# Convert each row to a tuple and add to the queue
for _, row in missing_runs_df.iterrows():
    combination = (row["Cell Type"], row["TF"], row["Negative Sampling Mode"], row["Cross Val Fold"])
    missing_queue.put(combination)

print(f"Loaded {missing_queue.qsize()} missing runs to process")

# Define worker function specifically for missing runs
def missing_worker():
    while not missing_queue.empty():
        cell_type, tf, neg_mode, set_idx = missing_queue.get()
        print(f"Processing missing run: {cell_type}, {tf}, {neg_mode}, {set_idx}")
        
        command = [
            "python", 
            "/data/home/natant/Negatives/TFBS_negatives/utils/train_model.py",
            "--datafolder", datafolder,
            "--TF", tf, 
            "--celltype", cell_type, 
            "--neg_mode", neg_mode, 
            "--devices", "1",
            "--cross_val_set", str(set_idx),
            "--learning_rate", "0.0001",
            "--n_blocks", "2",
            "--target_hsize", "128",
            "--batch_size", "256",
            "--output_dir", output_dir,
            "--early_stop_patience", "30",
            "--early_stop_metric", "AUROC",
            "--early_stop_mode", "max",
            "--group_name", f"{group_name}_recovery",
            "--test"
        ]
        subprocess.run(command)
        missing_queue.task_done()
        print(f"Completed missing run: {cell_type}, {tf}, {neg_mode}, {set_idx}")

# Create and start threads for processing missing runs
missing_threads = []
for _ in range(max_concurrent_models):
    t = Thread(target=missing_worker)
    t.daemon = True
    t.start()
    missing_threads.append(t)

# Wait for all threads to finish
for t in missing_threads:
    t.join()

print("All missing runs have been processed!")