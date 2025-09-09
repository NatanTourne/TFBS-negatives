import pandas as pd
import numpy as np
import h5torch
import py2bit
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        prog='GenerateDataset',
        description="Generate the h5torch dataset")

    parser.add_argument("--Gen_h5t", help="Create the h5torch dataset", action="store_true")

    parser.add_argument("-o", "--bed_location", help="Folder with the cell types specific bed files", default="/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC")
    parser.add_argument("-g", "--genome_file", help="location of the 2bit genome file (hg38)", default="/data/home/natant/plmBind/Data/hg38.2bit")
    parser.add_argument("-h5", "--h5t_location", help="location to save or get the h5t files")


    parser.add_argument("--Gen_dinucl_shuffled", help="add dinucleotide shuffled negs", action="store_true")
    parser.add_argument("--num_negs", help="number of negatives to generate FOR EACH POSITIVE", default=1)
    parser.add_argument("--sampl_dinucl_matched", help="sample dinucleotide matched negs", action="store_true")

    args = parser.parse_args()

    if args.Gen_h5t:
        print("Generating h5t dataset")
        create_h5torch(args.bed_location, args.genome_file, args.h5t_location)

    if args.Gen_dinucl_shuffled:
        print("Generating dinucleotide shuffled negatives")
        create_dinucl_shuffled_negatives(args.h5t_location, args.num_negs)

    if args.sampl_dinucl_matched:
        print("Generating dinucleotide matched negatives")
        create_dinucl_matched_negatives(args.h5t_location, args.num_negs)

def create_h5torch(bed_folder, genome_file, out_folder):
    # Ensure the output folder exists
    os.makedirs(out_folder, exist_ok=True)

    # Read the genome file
    tb = py2bit.open(genome_file)
    chromosomes = tb.chroms()
    chr_allow_list = [chrom for chrom in chromosomes.keys() if 'chrUn' not in chrom and 'chrM' not in chrom]

    # Initialize the chromosome allow list
    # chr_allow_list = list(np.arange(1, 23).astype(str)) + ["X", "Y"] 
    # chr_allow_list = ["chr" + c for c in chr_allow_list]


    for bed_file in tqdm(os.listdir(bed_folder)):
        if bed_file.endswith(".bed"):
            celltype = os.path.splitext(bed_file)[0]
            input_path = os.path.join(bed_folder, bed_file)
            output_path = os.path.join(out_folder, celltype.split("_")[0]+".h5t")
            
            print(f"Processing celltype: {celltype}")
            create_tf_presence_dataset(input_path, output_path, chr_allow_list, tb)

def create_tf_presence_dataset(bed_file, output_h5t_file, chr_allow_list, tb):
    chr_per_peaks = []
    pos_per_peaks = []
    len_per_peaks = []

    mapping = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}


    peaks = pd.read_csv(bed_file, sep="\t", header=None, names=["chr", "start", "end", "TF"])

    # Get unique TFs
    unique_tfs = peaks["TF"].unique()
    tf_to_index = {tf: idx for idx, tf in enumerate(unique_tfs)}

    # Initialize a binary presence array
    filtered_peaks = peaks[peaks["chr"].isin(chr_allow_list)]
    filtered_peaks = filtered_peaks.reset_index(drop=True)
    num_peaks = len(filtered_peaks)
    num_tfs = len(unique_tfs)
    tf_presence = np.zeros((num_tfs, num_peaks), dtype=int)

    # Iterate through each peak and mark its presence in the array
    chr_old = ""

    for peak_index, peak in tqdm(filtered_peaks.iterrows()):
        tf_index = tf_to_index[peak["TF"]]
        tf_presence[tf_index, peak_index] = 1

        # Handle overlapping peaks #! absurdly inefficient. But fuck it? It only needs to run once for each dataset. n² *4 (😬) 
        overlapping_peaks = filtered_peaks[
            (filtered_peaks["chr"] == peak["chr"]) & # same chrom
            (filtered_peaks["start"] < peak["end"]) & # overlap peak starts before our peak ends
            (filtered_peaks["end"] > peak["start"]) & # and ends after our peak starts! 
            (filtered_peaks.index != peak_index) # not the same peak
        ]
        for _, overlap_peak in overlapping_peaks.iterrows():
            overlap_tf_index = tf_to_index[overlap_peak["TF"]]
            tf_presence[overlap_tf_index, peak_index] = 2

        # register peak location
        middle = int((int(peak["start"]) + int(peak["end"])) / 2)
        chr_per_peaks.append(peak["chr"])
        pos_per_peaks.append(middle)
        len_per_peaks.append(int(peak["end"]) - int(peak["start"]))

    # Gather protein names in order they were filled in
    index_to_TF = {v : k for k, v in tf_to_index.items()}
    prot_names = np.array([index_to_TF[i] for i in range(len(index_to_TF))])

    # Gather sequence in int8 format:
    genome = {}
    for chr_ in chr_allow_list:
        genome[chr_] = np.array([mapping[bp] for bp in tb.sequence(chr_)], dtype="int8")

    # Save the dataset to an HDF5 file
    f = h5torch.File(output_h5t_file, "w")
    f.register(
        tf_presence, 
        "central", 
        mode="N-D", 
        dtype_save="int", 
        dtype_load="int"
    )

    f.register(
        prot_names.astype(bytes),
        axis=0,
        name = "prot_names",
        mode = "N-D",
        dtype_save = "bytes",
        dtype_load = "str"
    )
    f.register(
        np.array(chr_per_peaks).astype(bytes),
        axis=1,
        name="peak_ix_to_chr",
        mode = "N-D",
        dtype_save="bytes",
        dtype_load="str"
    )

    f.register(
        np.array(pos_per_peaks),
        axis=1,
        name="peak_ix_to_pos",
        mode = "N-D",
        dtype_save="int64",
        dtype_load="int64"
    )

    f.register(
        np.array(len_per_peaks),
        axis=1,
        name="peak_ix_to_len",
        mode = "N-D",
        dtype_save="int64",
        dtype_load="int64"
    )

    for k, v in genome.items():
        f.register(
            v,
            axis="unstructured",
            name=k,
            mode="N-D",
            dtype_save="int8",
            dtype_load="int8",
        )

    f.close()

def create_dinucl_shuffled_negatives(h5t_loc, num_negs):
    # Ensure the h5t_loc folder exists
    if not os.path.exists(h5t_loc):
        raise FileNotFoundError(f"The folder {h5t_loc} does not exist.")

    # Get all .h5t files in the folder
    h5t_files = [os.path.join(h5t_loc, file) for file in os.listdir(h5t_loc) if file.endswith(".h5t")]

    if not h5t_files:
        raise FileNotFoundError(f"No .h5t files found in the folder {h5t_loc}.")

    print(f"Found {len(h5t_files)} .h5t files in the folder {h5t_loc}.")

    mapping = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
    rev_mapping = {v : k for k, v in mapping.items()}

    for h5t_file in h5t_files:
        print(f"Processing file: {h5t_file}")
        with h5torch.File(h5t_file, "a") as f:
            genome = {k : f["unstructured"][k] for k in list(f["unstructured"]) if k.startswith("chr")}

            prot_names = [name.decode("utf-8") for name in f["0/prot_names"]]
            if "ATAC_peak" not in prot_names:
                raise ValueError("ATAC_peak not found in prot_names.")

            # Exclude "ATAC_peak" explicitly
            for i, TF in enumerate(tqdm(prot_names)):
                if TF == "ATAC_peak":
                    continue  # Skip ATAC_peak

                index = i
                pos_indices = np.where(f["central"][index, :] == 1)[0]

                dinucl_shuffled_seqs = []
                chr_list = []

                for j in tqdm(pos_indices):
                    chr = f["1/peak_ix_to_chr"][:][j].astype(str)
                    pos = f["1/peak_ix_to_pos"][:][j]
                    DNA_region_pos = genome[chr][pos - 50 : pos + 51]  #! Is this correct???
                    for i in range(num_negs):
                        shuffled = dinuclShuffle("".join([rev_mapping[l] for l in DNA_region_pos]))
                        encoded_shuffled = np.array([mapping[bp] for bp in shuffled], dtype="int8")
                        dinucl_shuffled_seqs.append(encoded_shuffled)
                        chr_list.append(chr)

                f.register(
                    np.stack(dinucl_shuffled_seqs),
                    axis="unstructured",
                    name=f"dinucl_{TF}_seqs",
                    mode="N-D",
                    dtype_save="int8",
                    dtype_load="int8",
                )

                f.register(
                    np.array(chr_list).astype(bytes),
                    axis="unstructured",
                    name=f"dinucl_{TF}_chrs",
                    mode="N-D",
                    dtype_save="bytes",
                    dtype_load="str",
                )

def create_dinucl_matched_negatives(h5t_loc, num_negs=1, disable_neg_file_creation=False):
  if not os.path.exists(h5t_loc):
      raise FileNotFoundError(f"The folder {h5t_loc} does not exist.")

  h5t_files = [os.path.join(h5t_loc, file) for file in os.listdir(h5t_loc) if file.endswith(".h5t")]

  if not disable_neg_file_creation:
    for h5t_file in h5t_files:
        celltype = os.path.splitext(os.path.basename(h5t_file))[0]
        print(f"Processing file: {h5t_file}")
        with h5torch.File(h5t_file, "r") as f:
            prot_names = [name.decode("utf-8") for name in f["0/prot_names"]]

            # Exclude "ATAC_peak" explicitly
            for i, TF in enumerate(tqdm(prot_names)):
                if TF == "ATAC_peak":
                    continue  # Skip ATAC_peak

                index = i
                pos_indices = np.where(f["central"][index, :] == 1)[0]

                # Write the positive sequences to a BED file
                output_path = os.path.join(h5t_loc, f"{celltype}_{TF}_positives_temp.bed")
                with open(output_path, "w") as bed_file:
                    for j in tqdm(pos_indices):
                        chr = f["1/peak_ix_to_chr"][:][j].astype(str)
                        pos = f["1/peak_ix_to_pos"][:][j]
                        start = pos - 50
                        end = pos + 51
                        bed_file.write(f"{chr}\t{start}\t{end}\t{TF}\n")


  # create negs
  r_script_path = "/data/home/natant/Negatives/TFBS_negatives/utils/ENCODE_chip/gkmsvm.R"
  gkmsvm_neg_sampling(r_script_path, h5t_loc, num_negs)

  #The bed files with the negative samples have the following naming scheme: # {celltype}_{TF}_negatives.bed
  negative_bed_files = [file for file in os.listdir(h5t_loc) if file.endswith("_negatives.bed")]

  for h5t_file in tqdm(h5t_files):
      celltype = os.path.splitext(os.path.basename(h5t_file))[0]
      print(f"Checking for negative bed files for cell type: {celltype}")
      
      matching_files = [file for file in negative_bed_files if file.startswith(celltype)]
      for bed_file_name in tqdm(matching_files):
          TF = "_".join(bed_file_name.split("_")[1:-3])  # Capture the full TF name, even with underscores
          bed_file_path = os.path.join(h5t_loc, bed_file_name)
          print(f"Processing negative bed file: {bed_file_path}")
          
          chromosomes = []
          centers = []
          lengths = []
          
          temp_file = pd.read_csv(bed_file_path, sep="\t", header=None, names=["chromosome", "start", "end"])
          chromosomes = temp_file["chromosome"].tolist()
          centers = ((temp_file["start"] + temp_file["end"]) // 2).tolist() # does this work????
          lengths = (temp_file["end"] - temp_file["start"]).tolist() #! the weird R script returns 100bp negatives for 101bp positives for some reason??????
          
          with h5torch.File(h5t_file, "a") as f:   
              f.register(
                  np.stack(centers),
                  axis="unstructured",
                  name=f"sampled_negs_{TF}_pos",
                  mode="N-D",
                  dtype_save="int64",
                  dtype_load="int64",
              )
              f.register(
                  np.array(chromosomes).astype(bytes),
                  axis="unstructured",
                  name=f"sampled_negs_{TF}_chr",
                  mode="N-D",
                  dtype_save="bytes",
                  dtype_load="str",
              )
              f.register(
                  np.array(lengths),
                  axis="unstructured",
                  name=f"sampled_negs_{TF}_len",
                  mode="N-D",
                  dtype_save="int8",
                  dtype_load="int8",
              )

def gkmsvm_neg_sampling(script_path, *args):
    # Convert all arguments to strings
    command = ["Rscript", script_path] + [str(arg) for arg in args]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("R script output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error while running R script:", e.stderr)
        raise

    




import sys,string,random
import subprocess

############################################################################################################### TODO: ANY PROBLEMS GOING FROM PYTHON 2 TO PYTHON 3
##### CODE MODIFIED FROM https://github.com/wassermanlab/BiasAway/blob/master/altschulEriksonDinuclShuffle.py 

def computeCountAndLists(s):
  #WARNING: Use of function count(s,'UU') returns 1 on word UUU
  #since it apparently counts only nonoverlapping words UU
  #For this reason, we work with the indices.

  #Initialize lists and mono- and dinucleotide dictionaries
  List = {} #List is a dictionary of lists
  List['A'] = []
  List['C'] = []
  List['G'] = []
  List['T'] = []
  List['N'] = []

  nuclList   = ["A","C","G","T","N"]
  s = s.upper()
  s = s.replace("T","T")
  nuclCnt = {}  #empty dictionary
  dinuclCnt = {}  #empty dictionary
  for x in nuclList:
    nuclCnt[x]=0
    dinuclCnt[x]={}
    for y in nuclList:
      dinuclCnt[x][y]=0

  #Compute count and lists
  nuclCnt[s[0]] = 1
  nuclTotal     = 1
  dinuclTotal   = 0
  for i in range(len(s)-1):
    x = s[i]; y = s[i+1]
    List[x].append( y )
    nuclCnt[y] += 1; nuclTotal  += 1
    dinuclCnt[x][y] += 1; dinuclTotal += 1
  assert (nuclTotal==len(s))
  assert (dinuclTotal==len(s)-1)
  return nuclCnt,dinuclCnt,List
 
 
def chooseEdge(x,dinuclCnt):
  numInList = 0
  for y in ['A','C','G','T','N']:
    numInList += dinuclCnt[x][y]
  z = random.random()
  denom=dinuclCnt[x]['A']+dinuclCnt[x]['C']+dinuclCnt[x]['G']+dinuclCnt[x]['T']+dinuclCnt[x]['N']
  numerator = dinuclCnt[x]['A']
  if z < float(numerator)/float(denom):
    dinuclCnt[x]['A'] -= 1
    return 'A'
  numerator += dinuclCnt[x]['C']
  if z < float(numerator)/float(denom):
    dinuclCnt[x]['C'] -= 1
    return 'C'
  numerator += dinuclCnt[x]['N']
  if z < float(numerator)/float(denom):
    dinuclCnt[x]['N'] -= 1
    return 'N'
  numerator += dinuclCnt[x]['G']
  if z < float(numerator)/float(denom):
    dinuclCnt[x]['G'] -= 1
    return 'G'
  dinuclCnt[x]['T'] -= 1
  return 'T'


def connectedToLast(edgeList,nuclList,lastCh):
  D = {}
  for x in nuclList: D[x]=0
  for edge in edgeList:
    a = edge[0]; b = edge[1]
    if b==lastCh: D[a]=1
  for i in range(2):
    for edge in edgeList:
      a = edge[0]; b = edge[1]
      if D[b]==1: D[a]=1
  ok = 0
  for x in nuclList:
    if x!=lastCh and D[x]==0: return 0
  return 1
 

def eulerian(s):
  nuclCnt,dinuclCnt,List = computeCountAndLists(s)
  #compute nucleotides appearing in s
  nuclList = []
  for x in ["A","C","G","T","N"]:
    if x in s: nuclList.append(x)
  #compute numInList[x] = number of dinucleotides beginning with x
  numInList = {}
  for x in nuclList:
    numInList[x]=0
    for y in nuclList:
      numInList[x] += dinuclCnt[x][y]
  #create dinucleotide shuffle L 
  firstCh = s[0]  #start with first letter of s
  lastCh  = s[-1]
  edgeList = []
  for x in nuclList:
    if x!= lastCh: edgeList.append( [x,chooseEdge(x,dinuclCnt)] )
  ok = connectedToLast(edgeList,nuclList,lastCh)
  return ok,edgeList,nuclList,lastCh


def shuffleEdgeList(L):
  n = len(L); barrier = n
  for i in range(n-1):
    z = int(random.random() * barrier)
    tmp = L[z]
    L[z]= L[barrier-1]
    L[barrier-1] = tmp
    barrier -= 1
  return L


def dinuclShuffle(s):
  ok = 0
  while not ok:
    ok,edgeList,nuclList,lastCh = eulerian(s)
  nuclCnt,dinuclCnt,List = computeCountAndLists(s)

  #remove last edges from each vertex list, shuffle, then add back
  #the removed edges at end of vertex lists.
  for [x,y] in edgeList: List[x].remove(y)
  for x in nuclList: shuffleEdgeList(List[x])
  for [x,y] in edgeList: List[x].append(y)

  #construct the eulerian path
  L = [s[0]]; prevCh = s[0]
  for i in range(len(s)-2):
    ch = List[prevCh][0] 
    L.append( ch )
    del List[prevCh][0]
    prevCh = ch
  L.append(s[-1])
  t = "".join(L)
  return t



if __name__ == "__main__":
    main()