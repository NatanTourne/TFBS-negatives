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
# Limit the number of concurrent processes
max_concurrent_models = 4
# Define the group name
group_name = "Review_sane_cross_cell_perf"


all_runs_df = pd.read_pickle('/data/home/natant/Negatives/TFBS_negatives/testing_ground/REVIEW_cross_cell_perf/20251218_all_cross_cell_runs_HQ.pkl')
run_tuples = [(row['file_path'], row['TF'], row['cv_split'], row['cellline_test']) 
              for _, row in all_runs_df.iterrows()]
queue = Queue()

# Populate the queue with cell_tf_neg_combinations
for combination in run_tuples:
    queue.put(combination)

# Function to process combinations from the queue
def worker():
    while not queue.empty():
        file_path,tf, cv, cell_type = queue.get()
        command = [
            "python", 
            "/data/home/natant/Negatives/TFBS_negatives/testing_ground/REVIEW_cross_cell_perf/Test_ckpt_HQ.py",
            "--ckpt_path", file_path,
            "--datafolder", datafolder,
            "--TF", tf, 
            "--celltype", cell_type, 
            "--devices", "2",
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