import subprocess
import itertools
import os
import h5torch

datafolder = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5"
# Initialize an empty dictionary to store cell types and their corresponding TFs
cell_types_dict = {}

# Iterate through all files in the datafolder
for file_name in os.listdir(datafolder):
    if file_name.endswith(".h5t"):  # Check if the file is an .h5t file
        cell_type = file_name.split(".")[0]  # Extract the cell type from the file name
        file_path = os.path.join(datafolder, file_name)
        
        # Open the .h5t file and extract TFs
        with h5torch.File(file_path, 'r') as h5_file:
            prot_names = h5_file["0/prot_names"][:]
            tf_list = [name.decode('utf-8') for name in prot_names if name.decode('utf-8') != "ATAC_peak"]
        
        # Add the cell type and its TFs to the dictionary
        cell_types_dict[cell_type] = tf_list




total_combos = sum(len(tfs) for tfs in cell_types_dict.values())
print(f"Total TF-cell type combinations: {total_combos}")

# Define the cell types
cell_types = ["MCF-7", "HCT-116"]  # Add more cell types as needed
# Check if all cell types are used
unused_cell_types = [ct for ct in cell_types if ct not in cell_types_dict]
if unused_cell_types:
    print(f"\033[91mWarning: The following cell types are not used in this debugging mode: {', '.join(unused_cell_types)}\033[0m")

negative_sampling_modes = ["dinucl_sampled", "dinucl_shuffled", "shuffled", "neighbors"]


# Generate combinations of cell types, their TFs, and negative sampling modes
cell_tf_neg_combinations = [
    (cell_type, tf, neg_mode, i)
    for cell_type, tfs in cell_types_dict.items()
    if cell_type in cell_types
    for tf in tfs
    for neg_mode in negative_sampling_modes
    for i in range(6)
]


from queue import Queue
from threading import Thread

# Limit the number of concurrent processes
max_concurrent_models = 8


# Create a queue to hold the combinations
queue = Queue()

# Populate the queue with cell_tf_neg_combinations
for combination in cell_tf_neg_combinations:
    queue.put(combination)

# Function to process combinations from the queue
def worker():
    while not queue.empty():
        cell_type, tf, neg_mode, set = queue.get()
        command = [
            "python", 
            "/data/home/natant/Negatives/TFBS_negatives/utils/train_model.py", 
            "--TF", tf, 
            "--celltype", cell_type, 
            "--neg_mode", neg_mode, 
            "--devices", "3",
            "--cross_val_set", str(set)
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

