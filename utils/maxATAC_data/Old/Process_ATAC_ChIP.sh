#!/bin/bash

OG_ATAC_DIR="/data/home/natant/plmBind/Data/LINKED_DATASET/maxATAC/ATAC_Signal_File"

# Directories containing the BED files
CHIP_DIR="/data/home/natant/plmBind/Data/LINKED_DATASET/maxATAC/ChIP_Binding_File_TEST" # Replace with the actual ChIP-seq directory path
ATAC_DIR="/data/home/natant/plmBind/Data/LINKED_DATASET/maxATAC/ATAC_Signal_File/Processed_C_0_5" # Replace with the actual ATAC-seq directory path


mkdir -p "$ATAC_DIR"

# Create ATAC_bed files
for i in $OG_ATAC_DIR/*.BedGraph; do
    [ -f "$i" ] || continue
    echo $i
    name=$(basename $i .BedGraph)
    fullpath="${ATAC_DIR}/${name}.bed"
    macs3 bdgpeakcall -i $i -o $fullpath -c 0.5
    #./bigWigToBedGraph $i $fullpath

done





# Output directory for merged files
OUTPUT_DIR="/data/home/natant/plmBind/Data/LINKED_DATASET/maxATAC/merged_chip_atac_files__C_0_5"
mkdir -p "$OUTPUT_DIR"


# Change to the ChIP-seq directory
cd "$CHIP_DIR" || { echo "ChIP-seq directory not found: $CHIP_DIR"; exit 1; }

# Get a list of unique cell types based on ChIP-seq files
cell_types=$(ls *.bed 2>/dev/null | awk -F'__' '{print $1}' | sort | uniq)

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
    concatenated_chip_file="${OUTPUT_DIR}/${cell}_chip_concatenated.bed"
    cat "${cell}__"*.bed | sort -k1,1 -k2,2n | awk 'BEGIN {OFS="\t"} {print $1,$2,$3,$4}' > "$concatenated_chip_file"

    
    # Merge the concatenated ChIP-seq peaks
    merged_chip_file="${OUTPUT_DIR}/${cell}_chip_merged.bed"
    merge_chip_peaks "$concatenated_chip_file" "$merged_chip_file"

    
    # Find the corresponding ATAC-seq file
    atac_file="${ATAC_DIR}/${cell}_RP20M_minmax_percentile99.bed"
    if [[ ! -f "$atac_file" ]]; then
        echo "ATAC-seq file not found for cell type $cell"
        continue
    fi

    atac_file_processed="${OUTPUT_DIR}/${cell}_atac.bed"
    tail -n +2 "$atac_file" | awk -F'\t' 'BEGIN {OFS="\t"} {print $1, $2, $3, 0}' > "$atac_file_processed"

#################################################################
    # Define intermediate and output files
    pos_file="${OUTPUT_DIR}/${cell}_pos.bed"
    neg_file="${OUTPUT_DIR}/${cell}_neg.bed"
    final_output_file="${OUTPUT_DIR}/${cell}_final_merged.bed"




    bedops --element-of 1 "$merged_chip_file" "$atac_file_processed" >  "$pos_file"

    bedops --not-element-of 1 "$atac_file_processed" "$merged_chip_file" > "$neg_file"
    
    # # Find non-intersecting ATAC-seq peaks and add a 0 in the fourth column
    # bedops --not-element-of 1 "$atac_file" "$merged_chip_file" | \
    #     awk 'BEGIN {OFS="\t"} {print $1,$2,$3,"0"}' > "$neg_file"
    
    # Combine intersecting and non-intersecting ATAC-seq peaks
    cat "$pos_file" "$neg_file" | sort -k1,1 -k2,2n > "$final_output_file"
    
    
    # Print a message indicating completion for the current cell type
    echo "Processed and merged files for cell type $cell into $final_output_file"
done
