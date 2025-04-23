#!/bin/bash

# Set paths to folders
BED_DIR="/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes"
ATAC_DIR="/data/home/natant/Negatives/Data/maxATAC/ATAC_Peaks_hg38"
OUTPUT_DIR="/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp_celltypes_ATAC"


# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop through BED files
for bed_file in "$BED_DIR"/*.narrowPeak; do
    # Extract basename and normalize cell type prefix
    base_name=$(basename "$bed_file" .narrowPeak)
    cell_type_prefix=$(echo "$base_name" | cut -d'_' -f1 | tr '[:upper:]' '[:lower:]' | tr -d '-')

    # Find a matching ATAC-seq file (case- and hyphen-insensitive)
    atac_file=$(ls "$ATAC_DIR"/*.narrowPeak 2>/dev/null | while read -r file; do
        # Normalize the ATAC file prefix the same way
        atac_base=$(basename "$file" .narrowPeak | cut -d'_' -f1 | tr '[:upper:]' '[:lower:]' | tr -d '-')
        if [[ "$atac_base" == "$cell_type_prefix" ]]; then
            echo "$file"
            break
        fi
    done)

    # Check if a matching ATAC file was found
    if [[ -n "$atac_file" ]]; then
        echo "Processing: $bed_file and $atac_file"

        # Modify ATAC file: Keep columns 1-3 and replace column 4 with "ATAC_peak"
        awk '{print $1"\t"$2"\t"$3"\tATAC_peak"}' "$atac_file" > "${atac_file}_filtered"

        # Merge TF BED and filtered ATAC file, then sort
        cat "$bed_file" "${atac_file}_filtered" | sort -k1,1 -k2,2n > "$OUTPUT_DIR/${base_name}_merged.bed"

        # Remove temporary filtered file
        rm "${atac_file}_filtered"

    else
        echo "No matching ATAC file for $bed_file (prefix: $cell_type_prefix), skipping."
    fi
done

echo "Merging completed. Merged files are in $OUTPUT_DIR"