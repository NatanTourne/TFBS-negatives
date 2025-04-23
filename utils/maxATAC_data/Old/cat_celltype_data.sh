#!/bin/bash

# FIRST RUN process_atac_seq.sh to create the bed files!

# Directory containing the BED files
DIR="/data/home/natant/plmBind/Data/LINKED_DATASET/maxATAC/ChIP_Binding_File_TEST" # Replace with the actual directory path

# Change to the directory
cd "$DIR" || { echo "Directory not found: $DIR"; exit 1; }

for i in $DIR/*.bed; do
    [ -f "$i" ] || continue
    echo $i
    name=$(basename $i .bed)
    TF=$(echo $name | cut -d "_" -f 3)
    cell=$(echo $name | cut -d "_" -f 1)
    fullpath="${DIR}/${name}.bed"
    echo $TF
    echo $cell
    awk -i inplace -v var="$TF" '{$4=var; print}' $i 
done

# Output directory for merged files
OUTPUT_DIR="merged_bed_files"
mkdir -p "$OUTPUT_DIR"

# Get a list of unique cell types
cell_types=$(ls *.bed 2>/dev/null | awk -F'__' '{print $1}' | sort | uniq)

# Check if any BED files are present
if [ -z "$cell_types" ]; then
  echo "No BED files found in the directory."
  exit 1
fi

# Function to merge overlapping peaks and concatenate TFs
merge_peaks() {
  local input_file=$1
  local output_file=$2
  
  # Use bedtools to merge peaks and retain the TF names
  bedtools merge -i "$input_file" -c 4 -o collapse > "$output_file"
}

# Loop through each cell type
for cell in $cell_types; do
    # Concatenate all files for the current cell type
    merged_file="${OUTPUT_DIR}/${cell}_merged.bed"
    
    # Preprocess files to ensure proper format
    cat "${cell}__"*.bed | sort -k1,1 -k2,2n | awk 'BEGIN {OFS="\t"} {print $1,$2,$3,$4}' > "$merged_file"
    
    # Temp file for storing merged peaks with TFs
    merged_peaks_file="${OUTPUT_DIR}/${cell}_merged_peaks.bed"
    
    # Merge overlapping peaks and list TFs
    merge_peaks "$merged_file" "$merged_peaks_file"
    
    # Print a message indicating completion for the current cell type
    echo "Processed and merged files for cell type $cell into $merged_peaks_file"
done
