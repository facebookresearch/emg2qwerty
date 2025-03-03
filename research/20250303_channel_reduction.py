# %%
from pathlib import Path

import h5py
import start_research

# %%

# Define the data directory
data_dir = Path("data/89335547_processed")

# Check if the directory exists
if not data_dir.exists():
    print(f"Directory {data_dir} does not exist")
    exit()

# Get all HDF5 files in the directory
hdf5_files = list(data_dir.glob("*.hdf5"))

if not hdf5_files:
    print(f"No HDF5 files found in {data_dir}")
    exit()

print(f"Found {len(hdf5_files)} HDF5 files in {data_dir}")
# %%


# Function to extract channel information from a file
def get_channel_info(file_path):
    with h5py.File(file_path, "r") as f:
        # Assuming the EMG data is stored in the 'emg2qwerty/timeseries' dataset
        if "emg2qwerty" in f and "timeseries" in f["emg2qwerty"]:
            timeseries = f["emg2qwerty"]["timeseries"]

            # Get the dtype fields to understand the structure
            dtype_fields = timeseries.dtype.fields

            # Check for EMG data fields
            right_channels = 0
            left_channels = 0

            if "emg_right" in dtype_fields:
                # Get the shape of the emg_right field for a single sample
                sample = timeseries[0]
                right_channels = sample["emg_right"].shape[0]

            if "emg_left" in dtype_fields:
                # Get the shape of the emg_left field for a single sample
                sample = timeseries[0]
                left_channels = sample["emg_left"].shape[0]

            return {
                "file": file_path.name,
                "right_channels": right_channels,
                "left_channels": left_channels,
                "total_samples": len(timeseries),
            }
        else:
            return {
                "file": file_path.name,
                "error": "Could not find emg2qwerty/timeseries in the file",
            }


# Process each file and collect channel information
channel_info = []
for file_path in hdf5_files:
    try:
        info = get_channel_info(file_path)
        channel_info.append(info)
        print(f"File: {info['file']}")
        if "error" in info:
            print(f"  Error: {info['error']}")
        else:
            print(f"  Right hand channels: {info['right_channels']}")
            print(f"  Left hand channels: {info['left_channels']}")
            print(f"  Total samples: {info['total_samples']}")
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")

# Summarize the findings
if channel_info:
    # Check if all files have the same channel structure
    right_channels = set(
        info.get("right_channels", 0) for info in channel_info if "error" not in info
    )
    left_channels = set(
        info.get("left_channels", 0) for info in channel_info if "error" not in info
    )

    print("\nSummary:")
    if len(right_channels) == 1 and len(left_channels) == 1:
        print("All files have consistent channel structure:")
        print(f"  Right hand channels: {next(iter(right_channels))}")
        print(f"  Left hand channels: {next(iter(left_channels))}")
    else:
        print("Files have inconsistent channel structure:")
        print(f"  Right hand channels: {right_channels}")
        print(f"  Left hand channels: {left_channels}")

# %%
