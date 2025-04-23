#!/bin/bash

# Define paths
INPUT_FOLDER="/data/home/natant/Negatives/Data/Encode690/filtered_hg19"  # Replace with your input folder path
OUTPUT_FOLDER="/data/home/natant/Negatives/Data/Encode690/filtered_hg38"  # Replace with your output folder path
CHAIN_FILE="/data/home/natant/Negatives/Data/Encode690/filtered_hg38/hg19ToHg38.over.chain.gz"  # Replace with your chain file path

# Ensure output folder exists
mkdir -p "$OUTPUT_FOLDER"


# Check if chain file exists
if [ ! -f "$CHAIN_FILE" ]; then
    echo "Error: Chain file not found at $CHAIN_FILE. Please download it from UCSC."
    exit 1
fi

# Process each BED file in the input folder
for file in "$INPUT_FOLDER"/*.narrowPeak; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        output_file="$OUTPUT_FOLDER/$filename"

        echo "Processing $filename..."

        # Run CrossMap
        CrossMap bed "$CHAIN_FILE" "$file" "$output_file"

        if [ $? -eq 0 ]; then
            echo "Successfully converted $filename."
        else
            echo "Error converting $filename. Check the input file format."
        fi
    else
        echo "No BED files found in $INPUT_FOLDER."
    fi
done

echo "All files processed."

