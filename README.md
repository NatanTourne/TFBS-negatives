# To Bind or Not to Bind: How Negative Sampling Shapes Transcription Factor Binding Site Prediction Performance

This repository accompanies the study investigating how different negative sampling strategies impact transcription factor (TF) binding site (TFBS) prediction. It provides:
- Data processing scripts to build per cell-line TF/ATAC peak datasets in HDF5 / h5torch format
- Multiple negative sampling regimes implemented as dataset classes
- A lightweight Enformer-inspired convolutional architecture (without transformer) for binary TF binding prediction
- Training utilities for standard and High-Quality (HQ) datasets, with optional limitation of positives for ablation / scaling experiments

## 1. Installation

Prerequisites: Python >=3.10 (setup requires >3.9), CUDA-capable GPUs recommended.

Clone and install (editable mode):
```
git clone <your_fork_or_repo_url> TFBS_negatives
cd TFBS_negatives
pip install -e .
```
Additional packages (not pinned in setup.py) you likely need:
```
pip install wandb h5torch einops transformers torchmetrics
```
(Optional) login to Weights & Biases for logging:
```
wandb login
```

## 2. Repository Layout (essentials)
- TFBS_negatives/models.py : EnformerConvStack + TFmodel / TFmodel_HQ (LightningModules)
- TFBS_negatives/data.py   : Dataset classes + Lightning DataModules
- utils/train_model.py                : Train with chosen negative sampling mode
- utils/train_model_HQ.py             : Train on High-Quality (ATAC filtered) dataset version
- utils/ENCODE_chip/                  : Scripts to build raw / intermediate datasets (ChIP + ATAC integration)

## 3. Data Generation Pipeline (script-grounded)
This section now mirrors the real executed steps in:
- utils/ENCODE_chip/Processing_steps.txt (standard set)


General directory placeholders (adapt to your paths):
RAW_HG19_DIR, LIFTOVER_HG38_DIR, FIXED101_DIR, CELLTYPE_DIR, CELLTYPE_ATAC_DIR, H5_DIR

Summary Table:
| Step | Description |
|------|-------------|
| 1 | Acquire raw ENCODE hg19 TF ChIP narrowPeaks |
| 2 | LiftOver & rename & metadata build (hg19→hg38) |
| 3 | Filter & restrict to cell types with ATAC + trim to 101 bp around summit (or BIG variant) |
| 4 | Build per-cell-type TF collections (optionally merge overlaps) OR keep separate |
| 5 | Append ATAC peaks (tagged ATAC_peak) per cell type |
| 6 | Create H5/h5t + (optional) precompute negative pools (dinucl shuffled / dinucl sampled) |
| 7 | Add cell-type contextual negative indices (for neg_mode=celltype) |


STEP 1. Acquire ChIP-seq datasets (hg19 uniform peaks)
- link: https://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/

STEP 2. LiftOver + renaming + metadata creation
- Goal: Convert coordinates to hg38 and produce metadata.csv with parsed attributes + new standardized filenames.
- Script: utils/ENCODE_chip/Processing_OG_ENCODE.py (invokes CrossMap bed). Chain file: hg19ToHg38.over.chain.gz.
- Inputs: RAW_HG19_DIR/*.narrowPeak(.gz), chain file.
- Outputs: LIFTOVER_HG38_DIR/*.narrowPeak (renamed), LIFTOVER_HG38_DIR/metadata.csv
- Notes / Pitfalls:
  * CrossMap must be installed and on PATH; script loops through files and builds filename_mapping from files.txt.
  * Any file lacking mapping is skipped (watch console warnings).
  * Ensure chain file path inside script matches your environment (variable chain_file).
  * metadata.csv later used to infer unique cell types (see Process_ChIP.sh / Making_* scripts).

STEP 3. Filter + harmonize peak length to 101 bp (centered on summit) & restrict to ATAC-supported cell types
- Goal: Produce fixed-length BEDs per TF-celltype; drop datasets whose cell type not present in ATAC collection (unless BIG variant).
- Scripts: utils/ENCODE_chip/Preparing_dataset.py
- Core logic (see adjust_peak_length in *_BIG.py): start + peak(summit offset) => peak_center; new interval = [center-50, center+51) giving 101 bp.
- Inputs: LIFTOVER_HG38_DIR/*.narrowPeak, (optionally) a mapping of acceptable cell types to ATAC dataset names.
- Outputs: FIXED101_DIR/<TF>__<CELL>__*.narrowPeak (or .bed) standardized to 101 bp.

STEP 4. Build per-cell-type TF collections
- Goal: Group per cell type all TF-specific fixed BEDs;
- Scripts: Making_Cell_Type_ChIP_datasets_no_merging.sh
- Inputs: FIXED101_DIR/<TF>__<CELL>* files, metadata.csv for cell type enumeration (awk in scripts).
- Outputs: CELLTYPE_DIR/ (structure varies). Typical: celltype-specific .narrowPeak/BED files with 4th col listing TF or comma-separated TFs.

STEP 5. Append ATAC peaks per cell type to enable High-Quality (HQ) negatives
- Goal: Introduce open chromatin intervals labeled ATAC_peak so each H5 can construct contextual negatives.
- Scripts: Adding_ATAC_peaks.sh
- Inputs: CELLTYPE_DIR/*.narrowPeak (or BED), ATAC_DIR/* (ATAC narrowPeak). Matching based on normalized cell prefix (lowercase, hyphen removed).
- Outputs: CELLTYPE_ATAC_DIR/*_merged.bed where TF peaks + ATAC peaks sorted; ATAC rows have 4th column = ATAC_peak.


STEP 6. Create H5 / h5torch dataset & (optionally) generate sequence-based negative pools
- Goal: Central consolidated file containing sequences, labels, annotations, and negative candidate pools.
- Primary Script: Create_H5_dataset.py (flags control generation)
  Key Flags:
    --Gen_h5t (produce .h5t via h5torch structure) OR default .h5
    --bed_location (CELLTYPE_ATAC_DIR for HQ, else CELLTYPE_DIR)
    --h5t_location (output directory for h5/h5t)
    --Gen_dinucl_shuffled (build dinucl-preserving shuffled seq sets per TF)
    --sampl_dinucl_matched (sample genome negatives matched on dinucleotide composition)
    --genomde_file (location of the 2bit genome file (hg38))
- Inputs: folder with the cell line specific bed files
- Outputs: H5_DIR/<celltype>.h5t (or .h5)
- Notes:
  * Sequence encoding stored under unstructured/chr* as int arrays (see data.py for A,T,C,G,N mapping).
  * Dinucleotide shuffled sequences stored as unstructured/dinucl_<TF>_{seqs,chrs}; sampled negatives as unstructured/sampled_negs_<TF>_{chr,pos}.
  * Keep consistent TF naming (lowercased, hyphen stripped in earlier scripts) or adjust data.py’s normalization.

STEP 7. Add cell-type contextual negative indices (neg_mode=celltype)
- Goal: For each TF, identify ATAC open regions in same cell type not bound by that TF and store index lists.
- Script: cell_line_sampling.py (run AFTER Step 6 if not already embedded).
- Inputs: H5_DIR/<celltype>.h5t with ATAC_peak entries available.
- Outputs: Adds unstructured/ct_sampled_<TF>_neg_indices datasets.

## 4. Negative Sampling Modes (neg_mode)
Implemented in data.py:
- neighbors        : Negatives sampled from sequence windows 101–200 bp away from each positive (same chrom).
- shuffled         : Negatives created by shuffling (permutation) of the positive sequences.
- dinucl_shuffled  : Precomputed dinucleotide-preserving shuffled sequences.
- dinucl_sampled   : Genomic sampled negatives based on similarity metrics.
- celltype         : Negatives selected as peaks (for another TF) in same cell line not bound by target TF.

Each mode defines a dataset returning dict samples with keys: "1/DNA_regions" (encoded ints) and "central" (0/1).

## 5. High-Quality (HQ) Dataset Variant
The HQ datasets are created by linking the ChIP-seq datasets with matching atac-seq data. Open atac-seq regions within the same cell line that don't overlap with peaks for a TF are taken as the negatives for that TF.

## 6. Quickstart: Train a Basic Model
Example using dinucleotide shuffled negatives:
```
python utils/train_model.py \
  --TF CTCF \
  --celltype GM12878 \
  --neg_mode dinucl_shuffled \
  --devices 0 \
  --batch_size 256 \
  --learning_rate 1e-4 \
  --n_blocks 4 \
  --target_hsize 256 \
  --early_stop_metric AUROC \
  --early_stop_mode max \
  --early_stop_patience 10 \
  --cross_val_set 0 \
  --group_name baseline \
  --test
```

## 7. Training on High-Quality Dataset
```
python utils/train_model_HQ.py \
  --TF CTCF \
  --celltype GM12878 \
  --devices 0 \
  --batch_size 256 \
  --learning_rate 1e-4 \
  --n_blocks 4 \
  --target_hsize 256 \
  --cross_val_set 0 \
  --group_name hq_baseline \
  --test
```


