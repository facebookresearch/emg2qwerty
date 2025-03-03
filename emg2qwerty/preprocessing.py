import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from sklearn.decomposition import PCA
from tqdm import tqdm


class DataPreprocessing:
    """Class for preprocessing EMG data, including dimensionality reduction and correlation
    analysis."""

    def __init__(self):
        """Initialize the DataPreprocessing class."""
        pass

    @staticmethod
    def explore_hdf5(item: Union[h5py.Group, h5py.Dataset, h5py.File], indent: int = 0) -> None:
        """Recursively explore and print the structure of an HDF5 file."""
        if isinstance(item, h5py.Group):
            print(" " * indent + f"GROUP: {item.name}")
            # Print attributes of the group
            if len(item.attrs) > 0:
                print(" " * (indent + 2) + "ATTRIBUTES:")
                for key, val in item.attrs.items():
                    print(" " * (indent + 4) + f"{key}: {val}")
            # Recursively explore group contents
            for key, val in item.items():
                DataPreprocessing.explore_hdf5(val, indent + 4)
        elif isinstance(item, h5py.Dataset):
            shape_str = str(item.shape)
            dtype_str = str(item.dtype)
            print(" " * indent + f"DATASET: {item.name}, Shape: {shape_str}, Type: {dtype_str}")
            # Print attributes of the dataset
            if len(item.attrs) > 0:
                print(" " * (indent + 2) + "ATTRIBUTES:")
                for key, val in item.attrs.items():
                    print(" " * (indent + 4) + f"{key}: {val}")
            # Print a few sample values if the dataset is small enough
            if len(item.shape) > 0 and item.shape[0] > 0 and np.prod(item.shape) < 10:
                print(" " * (indent + 4) + f"Values: {item[...]}")
        elif isinstance(item, h5py.File):
            print(" " * indent + f"FILE: {item.filename}")
            # Print attributes of the file
            if len(item.attrs) > 0:
                print(" " * (indent + 2) + "ATTRIBUTES:")
                for key, val in item.attrs.items():
                    print(" " * (indent + 4) + f"{key}: {val}")
            # Recursively explore file contents
            for key, val in item.items():
                DataPreprocessing.explore_hdf5(val, indent + 4)

    @staticmethod
    def open_hdf5_file(file_path: str) -> Optional[h5py.File]:
        """
        Open an HDF5 file and explore its contents.

        Args:
            file_path: Path to the HDF5 file

        Returns:
            Opened HDF5 file object or None if error occurs
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        try:
            with h5py.File(file_path, "r") as f:
                print(f"File: {file_path}")
                print("Structure:")
                DataPreprocessing.explore_hdf5(f)
                return f
        except Exception as e:
            print(f"Error opening file: {e}")
            return None

    @staticmethod
    def read_hdf5_dataset(file_path: str, dataset_path: str) -> Optional[np.ndarray]:
        """
        Read a specific dataset from an HDF5 file.

        Args:
            file_path: Path to the HDF5 file
            dataset_path: Path to the dataset within the HDF5 file

        Returns:
            Dataset as numpy array or None if error occurs
        """
        try:
            with h5py.File(file_path, "r") as f:
                if dataset_path not in f:
                    print(f"Dataset {dataset_path} not found in {file_path}")
                    return None

                dataset = f[dataset_path][...]
                print(
                    f"Dataset {dataset_path} loaded, shape: {dataset.shape}, type: {dataset.dtype}"
                )
                return np.array(dataset)
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return None

    def analyze_channel_correlations(
        self,
        emg_data: List[Dict],
        side: str = "both",
        method: str = "pearson",
        threshold: float = 0.8,
        plot: bool = True,
    ) -> Optional[Dict]:
        """
        Analyze correlations between EMG channels on the same side.

        Args:
            emg_data: Dataset containing EMG recordings
            side: 'right', 'left', or 'both' to specify which side(s) to analyze
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            threshold: Correlation threshold to highlight strong correlations
            plot: Whether to generate correlation heatmaps

        Returns:
            Dictionary containing correlation matrices and highly correlated channel pairs
        """
        if emg_data is None or len(emg_data) == 0:
            print("No EMG data to analyze")
            return None

        results = {}

        # Process right side if requested
        if side in ["right", "both"]:
            # Extract right EMG channels into a DataFrame
            n_channels_right = emg_data[0]["emg_right"].shape[0]
            right_data = np.array(
                [[entry["emg_right"][i] for i in range(n_channels_right)] for entry in emg_data]
            )

            # Create DataFrame with channel names
            right_df = pd.DataFrame(
                right_data,
                columns=[f"Right_Ch{i + 1}" for i in range(n_channels_right)],
            )

            # Calculate correlation matrix
            right_corr = right_df.corr(method=method)

            # Find highly correlated pairs (excluding self-correlations)
            high_corr_pairs_right = []
            for i in range(n_channels_right):
                for j in range(i + 1, n_channels_right):
                    corr_val = right_corr.iloc[i, j]
                    if abs(corr_val) >= threshold:
                        high_corr_pairs_right.append(
                            (right_corr.index[i], right_corr.columns[j], corr_val)
                        )

            # Sort by correlation strength
            high_corr_pairs_right.sort(key=lambda x: abs(x[2]), reverse=True)

            # Store results
            results["right"] = {
                "correlation_matrix": right_corr,
                "high_correlation_pairs": high_corr_pairs_right,
            }

            # Plot correlation heatmap
            if plot:
                plt.figure(figsize=(10, 8))
                mask = np.triu(np.ones_like(right_corr, dtype=bool))
                sns.heatmap(
                    right_corr,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    mask=mask,
                    vmin=-1,
                    vmax=1,
                    square=True,
                )
                plt.title(f"Right Hand EMG Channel Correlations ({method})")
                plt.tight_layout()
                plt.show()

        # Process left side if requested
        if side in ["left", "both"]:
            # Extract left EMG channels into a DataFrame
            n_channels_left = emg_data[0]["emg_left"].shape[0]
            left_data = np.array(
                [[entry["emg_left"][i] for i in range(n_channels_left)] for entry in emg_data]
            )

            # Create DataFrame with channel names
            left_df = pd.DataFrame(
                left_data, columns=[f"Left_Ch{i + 1}" for i in range(n_channels_left)]
            )

            # Calculate correlation matrix
            left_corr = left_df.corr(method=method)

            # Find highly correlated pairs (excluding self-correlations)
            high_corr_pairs_left = []
            for i in range(n_channels_left):
                for j in range(i + 1, n_channels_left):
                    corr_val = left_corr.iloc[i, j]
                    if abs(corr_val) >= threshold:
                        high_corr_pairs_left.append(
                            (left_corr.index[i], left_corr.columns[j], corr_val)
                        )

            # Sort by correlation strength
            high_corr_pairs_left.sort(key=lambda x: abs(x[2]), reverse=True)

            # Store results
            results["left"] = {
                "correlation_matrix": left_corr,
                "high_correlation_pairs": high_corr_pairs_left,
            }

            # Plot correlation heatmap
            if plot:
                plt.figure(figsize=(10, 8))
                mask = np.triu(np.ones_like(left_corr, dtype=bool))
                sns.heatmap(
                    left_corr,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    mask=mask,
                    vmin=-1,
                    vmax=1,
                    square=True,
                )
                plt.title(f"Left Hand EMG Channel Correlations ({method})")
                plt.tight_layout()
                plt.show()

        # Print summary of highly correlated channels
        if side in ["right", "both"] and results["right"]["high_correlation_pairs"]:
            print(f"\nHighly correlated right hand channels (|r| ≥ {threshold}):")
            for ch1, ch2, corr in results["right"]["high_correlation_pairs"]:
                print(f"  {ch1} ↔ {ch2}: {corr:.3f}")

        if side in ["left", "both"] and results["left"]["high_correlation_pairs"]:
            print(f"\nHighly correlated left hand channels (|r| ≥ {threshold}):")
            for ch1, ch2, corr in results["left"]["high_correlation_pairs"]:
                print(f"  {ch1} ↔ {ch2}: {corr:.3f}")

        return results

    def reduce_emg_channels(
        self,
        emg_data: List[Dict],
        correlation_threshold: float = 0.8,
        method: str = "pearson",
        reduction_method: str = "pca",
    ) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        """
        Reduce EMG data dimensionality by combining highly correlated channels.

        Args:
            emg_data: Dataset containing EMG recordings
            correlation_threshold: Threshold above which channels are considered redundant
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            reduction_method: Method to combine channels ('pca', 'average', or 'weighted_average')

        Returns:
            Tuple containing:
            - Reduced EMG dataset with combined channels
            - Dictionary with information about the reduction
        """
        if emg_data is None or len(emg_data) == 0:
            print("No EMG data to reduce")
            return None, None

        # First analyze correlations
        corr_results = self.analyze_channel_correlations(
            emg_data,
            side="both",
            method=method,
            threshold=correlation_threshold,
            plot=False,
        )
        if corr_results is None:
            return None, None

        reduction_info: dict[str, dict[str, Any]] = {
            "right": {
                "original_channels": 0,
                "channel_groups": [],
                "method": reduction_method,
            },
            "left": {
                "original_channels": 0,
                "channel_groups": [],
                "method": reduction_method,
            },
        }

        # Process right hand channels
        if "right" in corr_results:
            n_channels_right = emg_data[0]["emg_right"].shape[0]
            reduction_info["right"]["original_channels"] = n_channels_right

            # Extract channel data for correlation analysis
            right_data = np.array(
                [[entry["emg_right"][i] for i in range(n_channels_right)] for entry in emg_data]
            )

            # Group correlated channels
            channel_groups_right: list[set[int]] = []
            remaining_channels: set[int] = set(range(1, n_channels_right + 1))

            # Sort correlations by strength (highest first)
            high_corr_pairs = sorted(
                corr_results["right"]["high_correlation_pairs"],
                key=lambda x: abs(x[2]),
                reverse=True,
            )

            # Group highly correlated channels
            for ch1, ch2, corr_val in high_corr_pairs:
                ch1_idx = int(ch1.replace("Right_Ch", ""))
                ch2_idx = int(ch2.replace("Right_Ch", ""))

                # Find if either channel is already in a group
                group_found = False
                for group in channel_groups_right:
                    if ch1_idx in group or ch2_idx in group:
                        # Add the other channel to the existing group
                        if ch1_idx not in group:
                            group.add(ch1_idx)
                            remaining_channels.discard(ch1_idx)
                        if ch2_idx not in group:
                            group.add(ch2_idx)
                            remaining_channels.discard(ch2_idx)
                        group_found = True
                        break

                # If neither channel is in a group, create a new group
                if not group_found:
                    new_group = {ch1_idx, ch2_idx}
                    channel_groups_right.append(new_group)
                    remaining_channels.discard(ch1_idx)
                    remaining_channels.discard(ch2_idx)

            # Add remaining ungrouped channels as individual groups
            for ch in remaining_channels:
                channel_groups_right.append({ch})

            # Store the channel groups
            reduction_info["right"]["channel_groups"] = [
                sorted(list(group)) for group in channel_groups_right
            ]

        # Process left hand channels (similar to right hand)
        if "left" in corr_results:
            n_channels_left = emg_data[0]["emg_left"].shape[0]
            reduction_info["left"]["original_channels"] = n_channels_left

            # Extract channel data for correlation analysis
            left_data = np.array(
                [[entry["emg_left"][i] for i in range(n_channels_left)] for entry in emg_data]
            )

            # Group correlated channels
            channel_groups_left: list[set[int]] = []
            remaining_channels = set(range(1, n_channels_left + 1))

            # Sort correlations by strength (highest first)
            high_corr_pairs = sorted(
                corr_results["left"]["high_correlation_pairs"],
                key=lambda x: abs(x[2]),
                reverse=True,
            )

            # Group highly correlated channels
            for ch1, ch2, corr_val in high_corr_pairs:
                ch1_idx = int(ch1.replace("Left_Ch", ""))
                ch2_idx = int(ch2.replace("Left_Ch", ""))

                # Find if either channel is already in a group
                group_found = False
                for group in channel_groups_left:
                    if ch1_idx in group or ch2_idx in group:
                        # Add the other channel to the existing group
                        if ch1_idx not in group:
                            group.add(ch1_idx)
                            remaining_channels.discard(ch1_idx)
                        if ch2_idx not in group:
                            group.add(ch2_idx)
                            remaining_channels.discard(ch2_idx)
                        group_found = True
                        break

                # If neither channel is in a group, create a new group
                if not group_found:
                    new_group = {ch1_idx, ch2_idx}
                    channel_groups_left.append(new_group)
                    remaining_channels.discard(ch1_idx)
                    remaining_channels.discard(ch2_idx)

            # Add remaining ungrouped channels as individual groups
            for ch in remaining_channels:
                channel_groups_left.append({ch})

            # Store the channel groups
            reduction_info["left"]["channel_groups"] = [
                sorted(list(group)) for group in channel_groups_left
            ]

        # Create reduced dataset by combining channels within each group
        reduced_emg_data = []

        for entry in emg_data:
            reduced_right = []
            reduced_left = []

            # Process right hand channel groups
            for group in reduction_info["right"]["channel_groups"]:
                # Convert to 0-based indices for array access
                indices = [ch - 1 for ch in group]

                if len(indices) == 1:
                    # Single channel, just copy it
                    reduced_right.append(entry["emg_right"][indices[0]])
                else:
                    # Multiple channels to combine
                    channels_to_combine = entry["emg_right"][indices]

                    if reduction_method == "pca":
                        # Use PCA to reduce to first principal component
                        if len(channels_to_combine.shape) == 1:
                            # Only one sample, can't do PCA
                            reduced_right.append(np.mean(channels_to_combine))
                        else:
                            pca = PCA(n_components=1)
                            # Reshape for PCA if needed
                            reshaped = channels_to_combine.reshape(1, -1)
                            reduced_component = pca.fit_transform(reshaped.T)
                            reduced_right.append(reduced_component[0, 0])

                    elif reduction_method == "weighted_average":
                        # Calculate weights based on variance
                        variances = np.var(right_data[:, indices], axis=0)
                        weights = variances / np.sum(variances)
                        reduced_right.append(np.sum(channels_to_combine * weights))

                    else:  # Default to simple average
                        reduced_right.append(np.mean(channels_to_combine))

            # Process left hand channel groups (similar to right hand)
            for group in reduction_info["left"]["channel_groups"]:
                # Convert to 0-based indices for array access
                indices = [ch - 1 for ch in group]

                if len(indices) == 1:
                    # Single channel, just copy it
                    reduced_left.append(entry["emg_left"][indices[0]])
                else:
                    # Multiple channels to combine
                    channels_to_combine = entry["emg_left"][indices]

                    if reduction_method == "pca":
                        # Use PCA to reduce to first principal component
                        if len(channels_to_combine.shape) == 1:
                            # Only one sample, can't do PCA
                            reduced_left.append(np.mean(channels_to_combine))
                        else:
                            pca = PCA(n_components=1)
                            # Reshape for PCA if needed
                            reshaped = channels_to_combine.reshape(1, -1)
                            reduced_component = pca.fit_transform(reshaped.T)
                            reduced_left.append(reduced_component[0, 0])

                    elif reduction_method == "weighted_average":
                        # Calculate weights based on variance
                        variances = np.var(left_data[:, indices], axis=0)
                        weights = variances / np.sum(variances)
                        reduced_left.append(np.sum(channels_to_combine * weights))

                    else:  # Default to simple average
                        reduced_left.append(np.mean(channels_to_combine))

            # Create the reduced entry
            reduced_entry = {
                "time": entry["time"],
                "emg_right": np.array(reduced_right),
                "emg_left": np.array(reduced_left),
            }
            reduced_emg_data.append(reduced_entry)

        # Print summary
        print("\nDimensionality Reduction Summary:")
        print(f"Reduction method: {reduction_method}")
        print(
            f"Right hand: {n_channels_right} → "
            f"{len(reduction_info['right']['channel_groups'])} channels"
        )
        print("Channel groups:")
        for i, group in enumerate(reduction_info["right"]["channel_groups"]):
            if len(group) > 1:
                print(f"  Group {i + 1}: Channels {group} → Combined Channel {i + 1}")
            else:
                print(f"  Group {i + 1}: Channel {list(group)[0]} → Preserved as Channel {i + 1}")

        print(
            f"\nLeft hand: {n_channels_left} → "
            f"{len(reduction_info['left']['channel_groups'])} channels"
        )
        print("Channel groups:")
        for i, group in enumerate(reduction_info["left"]["channel_groups"]):
            if len(group) > 1:
                print(f"  Group {i + 1}: Channels {group} → Combined Channel {i + 1}")
            else:
                print(f"  Group {i + 1}: Channel {list(group)[0]} → Preserved as Channel {i + 1}")

        return reduced_emg_data, reduction_info

    def save_reduced_data(
        self,
        reduced_data: List[Dict],
        original_file_path: str,
        output_file_path: Optional[str] = None,
    ) -> None:
        """
        Save reduced EMG data to a new HDF5 file with the same format as the original.

        Args:
            reduced_data: Reduced EMG dataset to save
            original_file_path: Path to the original HDF5 file (to copy structure)
            output_file_path: Path where to save the new file. If None, will append '_reduced'
                            to the original filename
        """
        if output_file_path is None:
            # Generate output filename by adding '_reduced' before the extension
            base, ext = os.path.splitext(original_file_path)
            output_file_path = f"data/{base}_reduced{ext}"

        try:
            # Open both files
            with h5py.File(original_file_path, "r") as src, h5py.File(output_file_path, "w") as dst:
                # Copy all attributes from the source file
                for key, value in src.attrs.items():
                    dst.attrs[key] = value

                # Copy the basic structure but with reduced EMG data
                for key in src.keys():
                    if key == "emg2qwerty":  # Assuming this is the group containing the EMG data
                        # Create the group
                        emg_group = dst.create_group(key)

                        # Copy all attributes from the source group to preserve metadata like
                        # keystrokes
                        for attr_key, attr_value in src[key].attrs.items():
                            emg_group.attrs[attr_key] = attr_value

                        # Copy all datasets except timeseries
                        for subkey, item in src[key].items():
                            if subkey != "timeseries":
                                if isinstance(item, h5py.Dataset):
                                    dst[key].create_dataset(subkey, data=item[...])
                                elif isinstance(item, h5py.Group):
                                    # Recursively copy subgroups
                                    src.copy(f"{key}/{subkey}", dst[key])

                        # Create the reduced timeseries dataset with the same structure as the
                        # original but with fewer channels
                        n_samples = len(reduced_data)
                        n_right_channels = len(reduced_data[0]["emg_right"])
                        n_left_channels = len(reduced_data[0]["emg_left"])

                        # Create a new dtype for the reduced data
                        new_dtype = np.dtype(
                            [
                                ("emg_right", np.float32, (n_right_channels,)),
                                ("time", np.float64),
                                ("emg_left", np.float32, (n_left_channels,)),
                            ]
                        )

                        # Create the structured array
                        structured_data = np.empty(n_samples, dtype=new_dtype)

                        # Fill the structured array
                        for i, entry in enumerate(reduced_data):
                            structured_data[i]["emg_right"] = entry["emg_right"]
                            structured_data[i]["time"] = entry["time"]
                            structured_data[i]["emg_left"] = entry["emg_left"]

                        # Create the dataset
                        emg_group.create_dataset("timeseries", data=structured_data)
                    else:
                        # Copy other groups/datasets as is
                        if isinstance(src[key], h5py.Dataset):
                            dst.create_dataset(key, data=src[key][...])
                        else:
                            src.copy(key, dst)

            print(f"Reduced data saved to: {output_file_path}")

        except Exception as e:
            print(f"Error saving reduced data: {e}")


app = typer.Typer()


@app.command()
def process_file(
    file_path: Path = typer.Argument(..., help="Path to the HDF5 file to process"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory for processed files"),
    correlation_threshold: float = typer.Option(0.8, help="Threshold for channel correlation"),
    method: str = typer.Option("pearson", help="Correlation method: 'pearson' or 'spearman'"),
    reduction_method: str = typer.Option("pca", help="Reduction method: 'pca' or 'mean'"),
):
    """Process a single EMG file to reduce channels."""
    if not file_path.exists():
        typer.echo(f"Error: File {file_path} does not exist")
        raise typer.Exit(code=1)

    # Create output directory if not specified
    if output_dir is None:
        parent_dir = file_path.parent
        output_dir = Path(f"{parent_dir.name}_processed")

    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate output filename
    output_file = output_dir / file_path.name

    typer.echo(f"Processing {file_path}...")

    # Use the DataPreprocessing class to process the file
    try:
        # Create an instance of DataPreprocessing
        preprocessor = DataPreprocessing()

        # Load the data from the HDF5 file
        dataset = DataPreprocessing.read_hdf5_dataset(str(file_path), "emg2qwerty/timeseries")
        if dataset is None:
            typer.echo(f"Failed to read dataset from {file_path}")
            raise typer.Exit(code=1)

        # Convert the dataset to the expected format for reduce_emg_channels
        emg_data = []
        for entry in dataset:
            emg_entry = {
                "time": entry["time"],
                "emg_right": entry["emg_right"],
                "emg_left": entry["emg_left"],
            }
            emg_data.append(emg_entry)

        # Reduce channels
        reduced_data, reduction_info = preprocessor.reduce_emg_channels(
            emg_data,
            correlation_threshold=correlation_threshold,
            method=method,
            reduction_method=reduction_method,
        )

        if reduced_data is None:
            typer.echo(f"Failed to reduce channels in {file_path}")
            raise typer.Exit(code=1)

        # Save reduced data
        preprocessor.save_reduced_data(reduced_data, str(file_path), str(output_file))

        typer.echo(f"Successfully processed file: {output_file}")

    except Exception as e:
        typer.echo(f"Error processing file {file_path}: {e}")
        raise typer.Exit(code=1)


@app.command()
def process_directory(
    directory: Path = typer.Argument(..., help="Directory containing HDF5 files to process"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory for processed files"),
    correlation_threshold: float = typer.Option(0.8, help="Threshold for channel correlation"),
    method: str = typer.Option("pearson", help="Correlation method: 'pearson' or 'spearman'"),
    reduction_method: str = typer.Option("pca", help="Reduction method: 'pca' or 'mean'"),
    batch_size: int = typer.Option(
        1, help="Number of files to process at once before clearing memory"
    ),
):
    """Process all HDF5 files in a directory to reduce channels."""
    if not directory.exists() or not directory.is_dir():
        typer.echo(f"Error: Directory {directory} does not exist or is not a directory")
        raise typer.Exit(code=1)

    # Create output directory if not specified
    if output_dir is None:
        output_dir = Path(f"{directory.name}_processed")

    output_dir.mkdir(exist_ok=True, parents=True)

    # Find all HDF5 files
    hdf5_files = list(directory.glob("*.hdf5"))

    if not hdf5_files:
        typer.echo(f"No HDF5 files found in {directory}")
        raise typer.Exit(code=0)

    typer.echo(f"Found {len(hdf5_files)} HDF5 files to process")

    # Process files in batches to manage memory
    success_count = 0

    for i in range(0, len(hdf5_files), batch_size):
        batch = hdf5_files[i : i + batch_size]
        typer.echo(
            "Processing batch "
            f"{i // batch_size + 1}/"
            f"{(len(hdf5_files) + batch_size - 1) // batch_size}"
        )

        # Create a new preprocessor for each batch to avoid memory buildup
        preprocessor = DataPreprocessing()

        for file_path in tqdm(batch, desc="Processing files", unit="file"):
            output_file = output_dir / file_path.name
            typer.echo(f"Processing {file_path}...")

            if output_file.exists():
                typer.echo(f"Skipping {file_path.name} (already processed)")
                continue

            try:
                # Load the data from the HDF5 file
                dataset = None
                with h5py.File(str(file_path), "r") as f:
                    if "emg2qwerty/timeseries" not in f:
                        typer.echo(f"Dataset emg2qwerty/timeseries not found in {file_path}")
                        continue

                    # Load data in chunks to reduce memory usage
                    dataset = f["emg2qwerty/timeseries"][...]

                if dataset is None:
                    typer.echo(f"Failed to read dataset from {file_path}")
                    continue

                # Convert the dataset to the expected format for reduce_emg_channels
                emg_data = []
                for entry in dataset:
                    emg_entry = {
                        "time": entry["time"],
                        "emg_right": entry["emg_right"],
                        "emg_left": entry["emg_left"],
                    }
                    emg_data.append(emg_entry)

                # Clear dataset from memory
                del dataset
                import gc

                gc.collect()

                # Reduce channels
                reduced_data, reduction_info = preprocessor.reduce_emg_channels(
                    emg_data,
                    correlation_threshold=correlation_threshold,
                    method=method,
                    reduction_method=reduction_method,
                )

                # Clear emg_data from memory
                del emg_data
                gc.collect()

                if reduced_data is None:
                    typer.echo(f"Failed to reduce channels in {file_path}")
                    continue

                # Save reduced data
                preprocessor.save_reduced_data(reduced_data, str(file_path), str(output_file))

                # Clear variables to free memory
                del reduced_data
                del reduction_info
                gc.collect()

                success_count += 1

            except Exception as e:
                typer.echo(f"Error processing file {file_path}: {e}")
                import traceback

                typer.echo(traceback.format_exc())

        # Force garbage collection after each batch
        del preprocessor
        import gc

        gc.collect()

    typer.echo(f"Successfully processed {success_count} out of {len(hdf5_files)} files")


if __name__ == "__main__":
    app()
