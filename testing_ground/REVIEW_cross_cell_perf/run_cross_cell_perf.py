import pandas as pd
import subprocess
import itertools
import os
import h5torch
from queue import Queue
from threading import Thread
import csv


# data folder
datafolder = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr"
datafolder_celltype = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr copy"
# Limit the number of concurrent processes
max_concurrent_models = 4
# Define the group name
group_name = "Review_cross_cell_perf"


all_runs_df = pd.read_pickle('/data/home/natant/Negatives/testing_ground/REVIEW_cross_cell_perf/20251118_all_cross_cell_runs.pkl')
run_tuples = [(row['file_path'], row['TF'], row['negative_type'], row['cv_split'], row['cellline_test']) 
              for _, row in all_runs_df.iterrows()]
queue = Queue()

# Populate the queue with cell_tf_neg_combinations
for combination in run_tuples:
    queue.put(combination)

# Function to process combinations from the queue
def worker():
    while not queue.empty():
        file_path,tf, neg_mode, cv, cell_type = queue.get()
        if neg_mode == 'celltype':
            used_datafolder = datafolder_celltype
        else:
            used_datafolder = datafolder

        neg_mode = neg_mode.replace('-', '_')
        command = [
            "python", 
            "/data/home/natant/Negatives/testing_ground/REVIEW_cross_cell_perf/Test_ckpt.py",
            "--ckpt_path", file_path,
            "--datafolder", used_datafolder,
            "--TF", tf, 
            "--celltype", cell_type, 
            "--neg_mode", neg_mode, 
            "--devices", "1",
            "--cross_val_set", str(cv),
            "--batch_size", "256",
            "--group_name", group_name
        ]
        subprocess.run(command)
        queue.task_done()

# Create and start threads
threads = []
for _ in range(max_concurrent_models):
    t = Thread(target=worker)
    t.start()
    threads.append(t)

# Wait for all threads to finish
for t in threads:
    t.join()