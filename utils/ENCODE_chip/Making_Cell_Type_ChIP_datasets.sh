#!/bin/bash
CHIP_DIR="/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp"
OUTPUT_DIR="/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_celltype_merged"
# Change to the ChIP-seq directory
cd "$CHIP_DIR" || { echo "ChIP-seq directory not found: $CHIP_DIR"; exit 1; }

# Get a list of unique cell types based on ChIP-seq files
cell_types=$(ls *.narrowPeak 2>/dev/null | awk -F'__' '{print $1}' | sort | uniq)

# Check if any ChIP-seq BED files are present
if [ -z "$cell_types" ]; then
  echo "No ChIP-seq BED files found in the directory."
  exit 1
fi

# Function to merge overlapping peaks and concatenate TFs
merge_chip_peaks() {
  local input_file=$1
  local output_file=$2
  
  # Use bedtools to merge peaks and retain the TF names
  bedtools merge -i "$input_file" -c 4 -o collapse > "$output_file"
}

# Loop through each cell type
for cell in $cell_types; do
    echo "Processing cell type: $cell"
    
    # Concatenate all ChIP-seq files for the current cell type
    concatenated_chip_file="${OUTPUT_DIR}/${cell}_chip_concatenated.narrowPeak"
    cat "${cell}__"*.narrowPeak | sort -k1,1 -k2,2n | awk 'BEGIN {OFS="\t"} {print $1,$2,$3,$4}' > "$concatenated_chip_file"

    
    # Merge the concatenated ChIP-seq peaks
    merged_chip_file="${OUTPUT_DIR}/${cell}_chip_merged.narrowPeak"
    merge_chip_peaks "$concatenated_chip_file" "$merged_chip_file"

done
