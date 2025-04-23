import pandas as pd
import os
import shutil


ChIP_folder = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38"
metadata_file = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38/metadata.csv"
metadata_file_maxatac = "/data/home/natant/Negatives/Data/maxATAC/maxatac_chip.csv"

output_folder = "/data/home/natant/Negatives/Data/Encode690/ENCODE_hg38_subset_101bp"

# Load the metadata files
metadata_df = pd.read_csv(metadata_file)
metadata_maxatac_df = pd.read_csv(metadata_file_maxatac, sep='\t')

# Function to extract text before the first underscore, remove "-", and make it lowercase
def extract_tf_name(tf):
    return tf.split('_')[0].replace('-', '').lower()

# Apply the function to both metadata_maxatac_df and df
metadata_maxatac_df["tf_base"] = metadata_maxatac_df["tf"].apply(extract_tf_name)
metadata_df["antibody_base"] = metadata_df["antibody"].apply(extract_tf_name)

# Get unique base TFs from metadata_maxatac_df and df
maxatac_tfs_base = set(metadata_maxatac_df["tf_base"].unique())
df_tfs_base = set(metadata_df["antibody_base"].unique())

# Find common base TFs
common_tfs_base = maxatac_tfs_base.intersection(df_tfs_base)

# Find unique base TFs in metadata_maxatac_df
unique_maxatac_tfs_base = maxatac_tfs_base - df_tfs_base

# Find unique base TFs in df
unique_df_tfs_base = df_tfs_base - maxatac_tfs_base

# Print the results
print(f"Number of common base TFs: {len(common_tfs_base)}")
print(f"Number of unique base TFs in metadata_maxatac_df: {len(unique_maxatac_tfs_base)}")
print(f"Number of unique base TFs in df: {len(unique_df_tfs_base)}")

# Optionally, print the unique base TFs
print(f"Unique base TFs in metadata_maxatac_df: {unique_maxatac_tfs_base}")
print(f"Unique base TFs in df: {unique_df_tfs_base}")

# Create a list to store the matched TFs
matched_tfs_list = []
for tf_base in common_tfs_base:
    maxatac_tf = metadata_maxatac_df[metadata_maxatac_df["tf_base"] == tf_base]["tf"].values[0]
    df_tf = metadata_df[metadata_df["antibody_base"] == tf_base]["antibody"].values[0]
    matched_tfs_list.append({"maxatac_tf": maxatac_tf, "df_tf": df_tf})

# Convert the list to a DataFrame
matched_tfs = pd.DataFrame(matched_tfs_list)

# Display the matched TFs
print(matched_tfs)

# Function to normalize cell type names by making them lowercase and removing "-"
def normalize_cell_name(cell):
    return cell.replace('-', '').lower()

# Apply the function to both metadata_maxatac_df and df
metadata_maxatac_df["cell_base"] = metadata_maxatac_df["cell"].apply(normalize_cell_name)
metadata_df["cell_base"] = metadata_df["cell"].apply(normalize_cell_name)

# Get unique normalized cell types from metadata_maxatac_df and df
maxatac_cells_base = set(metadata_maxatac_df["cell_base"].unique())
df_cells_base = set(metadata_df["cell_base"].unique())

# Find common normalized cell types
common_cells_base = maxatac_cells_base.intersection(df_cells_base)

# Find unique normalized cell types in metadata_maxatac_df
unique_maxatac_cells_base = maxatac_cells_base - df_cells_base

# Find unique normalized cell types in df
unique_df_cells_base = df_cells_base - maxatac_cells_base

# Print the results
print(f"Number of common cell types: {len(common_cells_base)}")
print(f"Number of unique cell types in metadata_maxatac_df: {len(unique_maxatac_cells_base)}")
print(f"Number of unique cell types in df: {len(unique_df_cells_base)}")

# Optionally, print the unique cell types
print(f"Unique cell types in metadata_maxatac_df: {unique_maxatac_cells_base}")
print(f"Unique cell types in df: {unique_df_cells_base}")
# Create a list to store the matched cell types
matched_cells_list = []
for cell_base in common_cells_base:
    maxatac_cell = metadata_maxatac_df[metadata_maxatac_df["cell_base"] == cell_base]["cell"].values[0]
    df_cell = metadata_df[metadata_df["cell_base"] == cell_base]["cell"].values[0]
    matched_cells_list.append({"maxatac_cell": maxatac_cell, "df_cell": df_cell})

# Convert the list to a DataFrame
matched_cells = pd.DataFrame(matched_cells_list)

# Display the matched cell types
print(matched_cells)

# Get the original names of the unmatched cell types in metadata_maxatac_df
unmatched_maxatac_cells = metadata_maxatac_df[metadata_maxatac_df["cell_base"].isin(unique_maxatac_cells_base)]["cell"].unique()

# Get the original names of the unmatched cell types in df
unmatched_df_cells = metadata_df[metadata_df["cell_base"].isin(unique_df_cells_base)]["cell"].unique()

# Print the results
print(f"Unmatched cell types in metadata_maxatac_df: {unmatched_maxatac_cells}")
print(f"Unmatched cell types in df: {unmatched_df_cells}")


# Filter df to include only matched cells
matched_cells_set = set(matched_cells["df_cell"])
df_filtered_cells = metadata_df[metadata_df["cell"].isin(matched_cells_set)]

# Filter df_filtered_cells to include only matched TFs
matched_tfs_set = set(matched_tfs["df_tf"])
df_filtered = df_filtered_cells[df_filtered_cells["antibody"].isin(matched_tfs_set)]

# Display the filtered dataframe
df_filtered
# Add columns for corresponding celltype and tf name in the maxatac dataset
df_filtered = df_filtered.merge(matched_cells, left_on='cell', right_on='df_cell', how='left')
df_filtered = df_filtered.merge(matched_tfs, left_on='antibody', right_on='df_tf', how='left')


# Write df_filtered to the output folder as meta.csv
meta_output_path = os.path.join(output_folder, "metadata.csv")
df_filtered.to_csv(meta_output_path, sep='\t', index=False)


# Function to adjust the peak length to 101bp centered around the highest signal
def adjust_peak_length(df):
    df['peak_center'] = df['start'] + df['peak']
    df['start'] = df['peak_center'] - 50
    df['end'] = df['peak_center'] + 51
    return df.drop(columns=['peak_center'])

# Copy the files referenced in the filtered dataframe to the new folder
for filename in df_filtered["filename"]:
    if filename.endswith(".narrowPeak"):
        file_path = os.path.join(ChIP_folder, filename)
        df = pd.read_csv(file_path, sep='\t', header=None)
        df.columns = ['chrom', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak']

        tf_name = filename.split("__")[1]
        
        # Update the 'name' column with the TF and cell type
        df['name'] = tf_name
        
        # Adjust the peak length
        df = adjust_peak_length(df)
        
        # Save the modified file to the output folder
        output_path = os.path.join(output_folder, filename)
        df.to_csv(output_path, sep='\t', header=False, index=False)


