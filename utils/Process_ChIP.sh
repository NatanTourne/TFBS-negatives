#!/bin/bash

# Define the ChIP-seq directory path (replace with the actual path if needed)
CHIP_DIR="/data/home/natant/plmBind/Data/LINKED_DATASET/maxATAC/ChIP_Binding_File_TEST"
OUTPUT_DIR="/data/home/natant/Negatives/Data/Encode690/filtered_hg38_101bp_merged_celltypes"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Change to the ChIP-seq directory, exit with an error message if the directory is not found
cd "$CHIP_DIR" || { echo "ChIP-seq directory not found: $CHIP_DIR"; exit 1; }



# Extract unique cell types from the metadata.csv file
cell_types=$(awk -F',' 'NR>1 {print $6}' metadata.csv | sort | uniq)


# Function to merge overlapping peaks and concatenate TFs
merge_chip_peaks() {
  local input_file=$1
  local output_file=$2
  
  # Use bedtools to merge peaks and retain the TF names
  bedtools merge -i "$input_file" -c 4 -o collapse > "$output_file" 
  # bedtools merge: This is a command from the BEDTools suite, which is used for manipulating genomic intervals. The merge function specifically merges overlapping or adjacent intervals.
  # -i "$input_file": The -i option specifies the input file containing the intervals to be merged. $input_file is a variable that holds the path to this input BED file.
  # -c 4: The -c option specifies which column(s) to operate on during the merge. In this case, 4 refers to the fourth column of the BED file, which typically contains additional information such as the name of the transcription factor (TF).
  # -o collapse: The -o option specifies the operation to perform on the specified column(s). collapse means that the values in the fourth column for overlapping intervals will be concatenated together, separated by commas.
  # "$output_file": This redirects the output of the bedtools merge command to a file specified by the $output_file variable. This file will contain the merged intervals with the concatenated values from the fourth column.
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

    # Process the ATAC-seq file by removing the header and adding a 0 in the fourth column
    atac_file_processed="${OUTPUT_DIR}/${cell}_atac.bed"
    tail -n +2 "$atac_file" | awk -F'\t' 'BEGIN {OFS="\t"} {print $1, $2, $3, 0}' > "$atac_file_processed"

    #################################################################
    # Define intermediate and output files
    pos_file="${OUTPUT_DIR}/${cell}_pos.bed"
    neg_file="${OUTPUT_DIR}/${cell}_neg.bed"
    final_output_file="${OUTPUT_DIR}/${cell}_final_merged.bed"

    # Find intersecting ATAC-seq peaks and save to pos_file
    bedops --element-of 1 "$merged_chip_file" "$atac_file_processed" > "$pos_file"

    # Find non-intersecting ATAC-seq peaks and save to neg_file
    bedops --not-element-of 1 "$atac_file_processed" "$merged_chip_file" > "$neg_file"
    
    # Combine intersecting and non-intersecting ATAC-seq peaks
    cat "$pos_file" "$neg_file" | sort -k1,1 -k2,2n > "$final_output_file"
    
    # Print a message indicating completion for the current cell type
    echo "Processed and merged files for cell type $cell into $final_output_file"
done