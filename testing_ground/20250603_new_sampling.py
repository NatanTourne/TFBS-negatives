import pandas as pd
import numpy as np
import h5torch
import py2bit
from tqdm import tqdm
from tqdm import trange
import os



def create_ct_specific_sampled_negatives(h5t_loc):
    if not os.path.exists(h5t_loc):
      raise FileNotFoundError(f"The folder {h5t_loc} does not exist.")

    h5t_files = [os.path.join(h5t_loc, file) for file in os.listdir(h5t_loc) if file.endswith(".h5t")]

    for h5t_file in tqdm(h5t_files):
        print(f"Processing file: {h5t_file}")
        # Only process files for specific cell types
        celltypes = ["GM12878", "HepG2", "K562", "A549"]
        if not any(ct in h5t_file for ct in celltypes):
            print(f"Skipping file {h5t_file} (not in selected cell types)")
            continue
        
        with h5torch.File(h5t_file, 'a') as file:
            prot_names = file["0/prot_names"][:]

            # Check if all expected ct_sampled_{TF}_neg_indices already exist
            all_exist = True
            for TF in prot_names:
                if TF == b"ATAC_peak":
                    continue
                neg_name = f"ct_sampled_{TF.decode('utf-8')}_neg_indices"
                if neg_name not in file["unstructured"].keys():
                    all_exist = False
                    break

            if all_exist:
                print("All ct_sampled_{TF}_neg_indices datasets already exist. Skipping calculations for this file.")
                continue

            central = file["central"][:]
            atac_idx = np.where(prot_names == b"ATAC_peak")[0]
            mask_atac_peak = (central[atac_idx, :] == 1).any(axis=0)
            relevant_indices = np.where(~mask_atac_peak)[0]

            # Get sequences for the indices
            print("Retrieving sequences...")
            sequences = get_sequence(file, relevant_indices, rev_map=True)

            # Calculate dinucleotide frequency vectors for each sequence
            print("Calculating dinucleotide frequency vectors...")
            dinuc_vectors = np.array([dinuc_freq_vector(seq) for seq in sequences])

            # Create a DataFrame: rows = indices, columns = dinucleotide names
            DINUC_LIST = [
        "AA","AC","AG","AT",
        "CA","CC","CG","CT",
        "GA","GC","GG","GT",
        "TA","TC","TG","TT",]

            # Map each dinucleotide to an index 0..15 for quick lookup:
            DINUC_IDX = {dinuc: i for i, dinuc in enumerate(DINUC_LIST)}
            df_dinuc = pd.DataFrame(dinuc_vectors, index=relevant_indices, columns=DINUC_LIST)

            for TF in tqdm(prot_names):
                if TF == b"ATAC_peak":
                    continue
                print(f"Processing TF: {TF.decode('utf-8')}")
                # Check if the sampled negatives dataset already exists
                neg_name = f"ct_sampled_{TF.decode('utf-8')}_neg_indices"

                if neg_name in file["unstructured"].keys():
                    print(f"Warning: {neg_name} already exists in file. Skipping TF {TF.decode('utf-8')}.")
                    continue

                TF_idx = np.where(prot_names == TF)[0]
                tf_row = central[TF_idx, :]
                positive_indices = np.where(tf_row == 1)[1]
                negative_indices = np.where((tf_row == 0) & (~mask_atac_peak))[1] # so negative and not an ATAC peak!
                
                n_pos = len(positive_indices)
                n_neg = len(negative_indices)
                if n_neg == n_pos:
                    print(f"Warning: Number of negatives equals number of positives for TF {TF.decode('utf-8')}. Taking all negatives as sampled negatives.")
                    sampled_negatives = negative_indices
                else:
                    print(f"Number of positives: {n_pos}, Number of negatives: {n_neg}")
                    print("Sampling negatives...")
                    sampled_negatives = np.array(greedy_match_negatives_from_indices(df_dinuc, positive_indices, negative_indices))

                TF = TF.decode("utf-8")
                print(f"Saving sampled negatives for TF: {TF}")
                file.register(
                        sampled_negatives,
                        axis="unstructured",
                        name=f"ct_sampled_{TF}_neg_indices",
                        mode="N-D",
                        dtype_save="int64",
                        dtype_load="int64",
                    )



def get_sequence(f, indices, rev_map=True):
    """
    Get the sequence for a given index from the H5torch file.
    """
    mapping = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
    rev_mapping = {v : k for k, v in mapping.items()}
    genome = {k : f["unstructured"][k] for k in list(f["unstructured"]) if k.startswith("chr")}
    seqs = []
    for j in tqdm(indices): 
        chr = f["1/peak_ix_to_chr"][:][j].astype(str)
        pos = f["1/peak_ix_to_pos"][:][j]
        seq = genome[chr][pos-50:pos+51]
        if rev_map:
            seq = "".join([rev_mapping[bp] for bp in seq])
        seqs.append(seq)
    seqs = np.array(seqs)
    return seqs



def dinuc_freq_vector(seq: str) -> np.ndarray:
    """
    Given a 101‐bp string `seq`, compute a length‐16 vector of dinucleotide frequencies.
    We slide a window of length 2 along positions 0→1, 1→2, …, 99→100 (i.e. 100 windows total).
    Returns a numpy array of shape (16,), where each entry = (count of that dinuc) / 100.
    """
    DINUC_LIST = [
    "AA","AC","AG","AT",
    "CA","CC","CG","CT",
    "GA","GC","GG","GT",
    "TA","TC","TG","TT",
]

    # Map each dinucleotide to an index 0..15 for quick lookup:
    DINUC_IDX = {dinuc: i for i, dinuc in enumerate(DINUC_LIST)}

    seq = seq.upper()
    cnt = np.zeros(16, dtype=float)
    total_windows = len(seq) - 1  # should be 100 for a 101 bp input

    # Count dinucleotides
    for i in range(total_windows):
        di = seq[i:i+2]
        if di in DINUC_IDX:
            cnt[DINUC_IDX[di]] += 1
        else:
            # If there are ambiguous bases (e.g. 'N'), you could either skip or distribute arbitrarily.
            # Here, we simply skip any window containing non‐ACGT.
            pass

    # Convert counts → frequencies (divide by total_windows)
    if total_windows > 0:
        cnt /= float(total_windows)
    return cnt

def greedy_match_negatives_from_indices(dinuc_matrix,
                                        pos_indices,
                                        neg_indices):

    # Number of positives and negatives
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    if n_pos == 0:
        raise ValueError("pos_indices is empty; no positives to match.")
    if n_neg < n_pos:
        raise ValueError(
            f"Not enough negatives ({n_neg}) to match {n_pos} positives."
        )

    # Build an array of negative‐vectors keyed in the same order as neg_indices
    neg_vecs = dinuc_matrix.loc[neg_indices]   # shape = (n_neg, 16)

    # We will keep track of which negatives remain available by
    # maintaining a Python list of remaining_neg_indices and a parallel array remaining_neg_vecs.
    remaining_neg_indices = neg_indices.copy()
    remaining_neg_vecs = neg_vecs.copy()  # shape = (n_neg, 16)

    matched_negatives = []  # will end up length n_pos

    for p in tqdm(pos_indices):
        p_vec = dinuc_matrix.loc[p]            # shape = (16,)
        # Compute distance from p_vec to each available negative‐vector
        #   diff shape = (current_pool_size, 16)
        diff = remaining_neg_vecs - p_vec
        dists = np.linalg.norm(diff, axis=1)   # Euclidean, shape = (current_pool_size,)

        # Find argmin among the still‐available negatives
        best_j = np.argmin(dists)
        best_neg_idx = remaining_neg_indices[best_j]

        # Record it and remove from the pool
        matched_negatives.append(best_neg_idx)
        remaining_neg_indices = np.delete(remaining_neg_indices, best_j)
        remaining_neg_vecs = remaining_neg_vecs.drop(remaining_neg_vecs.index[best_j])

    return matched_negatives


create_ct_specific_sampled_negatives("/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC_H5_all_chr copy")