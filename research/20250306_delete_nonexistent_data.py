# %%
import os

import pandas as pd
import start_research

# %%
# Path to the dataset directory and metadata file
dataset_dir = "mnt/dataset"
metadata_file = "mnt/dataset/metadata.csv"

# Read the metadata file
metadata = pd.read_csv(metadata_file)
print(f"Original metadata rows: {len(metadata)}")
# %%
# Get list of files in the dataset directory
dataset_files = os.listdir(dataset_dir)
# Remove metadata.csv itself from the list
if "metadata.csv" in dataset_files:
    dataset_files.remove("metadata.csv")

# Extract the base filenames without .hdf5 extension for comparison
file_bases = [os.path.splitext(f)[0] for f in dataset_files]


# %%
# Function to check if a session exists in the file list
def session_exists(session, file_list):
    # Check if session name is a file
    if session in file_list:
        return True

    # Check if any file contains the session string
    for f in file_list:
        if session in f:
            return True

    return False


# %%
# Filter to keep only rows where the session exists as a file
filtered_metadata = metadata[metadata["session"].apply(lambda x: session_exists(x, dataset_files))]

# Save the filtered metadata back to the same file
filtered_metadata.to_csv(metadata_file, index=False)

print(f"Filtered metadata rows: {len(filtered_metadata)}")
print(f"Removed {len(metadata) - len(filtered_metadata)} rows")

# %%
