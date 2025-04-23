import pandas as pd
import os
import subprocess

# Define the path to your metadata file
input_folder = "/data/home/natant/Negatives/Data/Encode690/wgEncodeAwgTfbsUniform/"
output_folder = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38"
metadata_file = "/data/home/natant/Negatives/Data/Encode690/wgEncodeAwgTfbsUniform/files.txt"
chain_file = "/data/home/natant/Negatives/TFBS-negatives/utils/ENCODE_chip/hg19ToHg38.over.chain.gz"


# Initialize a list to store parsed metadata
data = []

# Read the file line by line
with open(metadata_file, "r") as file:
    for line in file:
        # Split the filename from the metadata
        filename, metadata = line.split(".gz")
        filename += ".gz"
        
        # Split metadata into key-value pairs
        metadata_dict = dict(
            item.split("=") for item in metadata.strip().split("; ") if "=" in item
        )
        
        # Add the filename to the metadata
        metadata_dict["filename"] = filename
        
        # Append to the list
        data.append(metadata_dict)

# Create a DataFrame from the parsed data
df = pd.DataFrame(data)

# Display the DataFrame
# Rename the filename column to old_filename
df = df.rename(columns={"filename": "old_filename"})

# Create a new filename column with the specified format
df["filename"] = df.apply(
    lambda row: f"{row['cell']}__{row['antibody']}__{row['treatment']}.narrowPeak", axis=1
)


# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Check if chain file exists
if not os.path.isfile(chain_file):
    raise FileNotFoundError(f"Error: Chain file not found at {chain_file}. Please download it from UCSC.")

# Process each BED file in the input folder
# Create a dictionary to map old filenames to new filenames
filename_mapping = dict(zip(df["old_filename"].str.replace(".gz", ""), df["filename"]))

for filename in os.listdir(input_folder):
    if filename.endswith(".narrowPeak"):
        input_file = os.path.join(input_folder, filename)
        
        # Get the new filename from the mapping
        new_filename = filename_mapping.get(filename)
        if new_filename:
            output_file = os.path.join(output_folder, new_filename)
        else:
            print(f"Warning: No mapping found for {filename}. Skipping.")
            continue

        print(f"Processing {filename}...")

        # Run CrossMap
        result = subprocess.run(["CrossMap", "bed", chain_file, input_file, output_file], capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully converted {filename}.")
        else:
            print(f"Error converting {filename}. Check the input file format.")
            print(result.stderr)
    else:
        print(f"No BED files found in {input_folder}.")


print("All files processed.")

# Write the DataFrame to a CSV file in the output folder
metadata_output_file = os.path.join(output_folder, "metadata.csv")
df.to_csv(metadata_output_file, index=False)
print(f"Metadata written to {metadata_output_file}.")