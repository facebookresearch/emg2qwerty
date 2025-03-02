# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import start_research
# %%
def explore_hdf5(item, indent=0):
    """Recursively explore and print the structure of an HDF5 file."""
    if isinstance(item, h5py.Group):
        print(" " * indent + f"GROUP: {item.name}")
        for key, val in item.items():
            explore_hdf5(val, indent + 4)
    elif isinstance(item, h5py.Dataset):
        shape_str = str(item.shape)
        dtype_str = str(item.dtype)
        print(" " * indent + f"DATASET: {item.name}, Shape: {shape_str}, Type: {dtype_str}")
        # Print a few sample values if the dataset is small enough
        if len(item.shape) > 0 and item.shape[0] > 0 and np.prod(item.shape) < 10:
            print(" " * (indent + 4) + f"Values: {item[...]}")

# Function to open and explore an HDF5 file
def open_hdf5_file(file_path):
    """Open an HDF5 file and explore its contents."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"File: {file_path}")
            print("Structure:")
            explore_hdf5(f)
            return f
    except Exception as e:
        print(f"Error opening file: {e}")
        return None

# Function to read a dataset from an HDF5 file
def read_hdf5_dataset(file_path, dataset_path):
    """Read a specific dataset from an HDF5 file."""
    try:
        with h5py.File(file_path, 'r') as f:
            if dataset_path not in f:
                print(f"Dataset {dataset_path} not found in {file_path}")
                return None
            
            dataset = f[dataset_path][...]
            print(f"Dataset {dataset_path} loaded, shape: {dataset.shape}, type: {dataset.dtype}")
            return dataset
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return None
    
def plot_emg_data(emg_data, n_samples=500):
    """Plot EMG data with multiple channels in separate subplots.
    
    Args:
        emg_data: A dataset where each element is a dictionary with keys 
                 'time', 'emg_right', and 'emg_left'.
    """
    if emg_data is None or len(emg_data) == 0:
        print("No EMG data to plot")
        return
    
    # Extract the first 500 measurements
    n_samples = min(n_samples, len(emg_data))
    
    # Extract time data for x-axis
    time_data = np.array([entry['time'] for entry in emg_data[:n_samples]])
    
    # Get the number of channels in each hand's EMG data
    n_channels_right = emg_data[0]['emg_right'].shape[0]
    n_channels_left = emg_data[0]['emg_left'].shape[0]
    total_channels = n_channels_right + n_channels_left
    
    # Create a figure with subplots for each channel
    fig, axes = plt.subplots(total_channels, 1, figsize=(15, 2*total_channels), sharex=True)
    
    # Collect all channel data for normalization
    right_channel_data = []
    left_channel_data = []
    
    # Collect right hand channel data
    for i in range(n_channels_right):
        right_channel_data.append(np.array([entry['emg_right'][i] for entry in emg_data[:n_samples]]))
    
    # Collect left hand channel data
    for i in range(n_channels_left):
        left_channel_data.append(np.array([entry['emg_left'][i] for entry in emg_data[:n_samples]]))
    
    # Find min and max for right hand channels
    right_min = min(np.min(data) for data in right_channel_data) if right_channel_data else 0
    right_max = max(np.max(data) for data in right_channel_data) if right_channel_data else 1
    
    # Find min and max for left hand channels
    left_min = min(np.min(data) for data in left_channel_data) if left_channel_data else 0
    left_max = max(np.max(data) for data in left_channel_data) if left_channel_data else 1
    
    # Add a small buffer to the limits
    right_margin = (right_max - right_min) * 0.05
    right_y_min = right_min - right_margin
    right_y_max = right_max + right_margin
    
    left_margin = (left_max - left_min) * 0.05
    left_y_min = left_min - left_margin
    left_y_max = left_max + left_margin
    
    # Plot right hand EMG channels
    for i in range(n_channels_right):
        channel_data = right_channel_data[i]
        axes[i].plot(time_data, channel_data)
        axes[i].set_ylabel(f"Right Ch {i+1}")
        axes[i].set_ylim(right_y_min, right_y_max)
        axes[i].grid(True)
    
    # Plot left hand EMG channels
    for i in range(n_channels_left):
        channel_data = left_channel_data[i]
        axes[i + n_channels_right].plot(time_data, channel_data)
        axes[i + n_channels_right].set_ylabel(f"Left Ch {i+1}")
        axes[i + n_channels_right].set_ylim(left_y_min, left_y_max)
        axes[i + n_channels_right].grid(True)
    
    # Set common x-axis label
    axes[-1].set_xlabel("Time (s)")
    
    # Add a title to the figure
    fig.suptitle("EMG Data - First 500 Measurements Across All Channels", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for the title
    plt.show()
# %%
file_path = "data/89335547/2021-06-02-1622679967-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5"
# %%
open_hdf5_file(file_path)
# %%
emg_data = read_hdf5_dataset(file_path, "/emg2qwerty/timeseries")
# %%
emg_data.shape
# %%
type(emg_data[0])
# %%
if emg_data is not None:
    plt.figure(figsize=(15, 6))
        
    # Plot the full timeseries
    plt.subplot(2, 1, 1)
    plt.plot(emg_data)
    plt.title(f"Full EMG Timeseries: /emg2qwerty/timeseries")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    
    # Plot a smaller segment to see details
    plt.subplot(2, 1, 2)
    segment_length = min(10000, len(emg_data))  # Show first 10,000 points or less
    plt.plot(emg_data[:segment_length])
    plt.title(f"First {segment_length} samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.show()

# %%
emg_data[0]["emg_right"]
# %%
emg_data[0]["emg_left"]
# %%
emg_data[0]["emg_right"].shape
# %%
emg_data[0]["time"]
# %%
plot_emg_data(emg_data, n_samples=2000)
# %%
def analyze_channel_correlations(emg_data, side='both', method='pearson', threshold=0.8, plot=True):
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
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr, kendalltau
    import seaborn as sns
    
    if emg_data is None or len(emg_data) == 0:
        print("No EMG data to analyze")
        return None
    
    results = {}
    
    # Process right side if requested
    if side in ['right', 'both']:
        # Extract right EMG channels into a DataFrame
        n_channels_right = emg_data[0]['emg_right'].shape[0]
        right_data = np.array([[entry['emg_right'][i] for i in range(n_channels_right)] 
                              for entry in emg_data])
        
        # Create DataFrame with channel names
        right_df = pd.DataFrame(right_data, 
                               columns=[f'Right_Ch{i+1}' for i in range(n_channels_right)])
        
        # Calculate correlation matrix
        right_corr = right_df.corr(method=method)
        
        # Find highly correlated pairs (excluding self-correlations)
        high_corr_pairs_right = []
        for i in range(n_channels_right):
            for j in range(i+1, n_channels_right):
                corr_val = right_corr.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs_right.append((
                        right_corr.index[i], 
                        right_corr.columns[j], 
                        corr_val
                    ))
        
        # Sort by correlation strength
        high_corr_pairs_right.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Store results
        results['right'] = {
            'correlation_matrix': right_corr,
            'high_correlation_pairs': high_corr_pairs_right
        }
        
        # Plot correlation heatmap
        if plot:
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(right_corr, dtype=bool))
            sns.heatmap(right_corr, annot=True, fmt=".2f", cmap="coolwarm", 
                       mask=mask, vmin=-1, vmax=1, square=True)
            plt.title(f"Right Hand EMG Channel Correlations ({method})")
            plt.tight_layout()
            plt.show()
    
    # Process left side if requested
    if side in ['left', 'both']:
        # Extract left EMG channels into a DataFrame
        n_channels_left = emg_data[0]['emg_left'].shape[0]
        left_data = np.array([[entry['emg_left'][i] for i in range(n_channels_left)] 
                             for entry in emg_data])
        
        # Create DataFrame with channel names
        left_df = pd.DataFrame(left_data, 
                              columns=[f'Left_Ch{i+1}' for i in range(n_channels_left)])
        
        # Calculate correlation matrix
        left_corr = left_df.corr(method=method)
        
        # Find highly correlated pairs (excluding self-correlations)
        high_corr_pairs_left = []
        for i in range(n_channels_left):
            for j in range(i+1, n_channels_left):
                corr_val = left_corr.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs_left.append((
                        left_corr.index[i], 
                        left_corr.columns[j], 
                        corr_val
                    ))
        
        # Sort by correlation strength
        high_corr_pairs_left.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Store results
        results['left'] = {
            'correlation_matrix': left_corr,
            'high_correlation_pairs': high_corr_pairs_left
        }
        
        # Plot correlation heatmap
        if plot:
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(left_corr, dtype=bool))
            sns.heatmap(left_corr, annot=True, fmt=".2f", cmap="coolwarm", 
                       mask=mask, vmin=-1, vmax=1, square=True)
            plt.title(f"Left Hand EMG Channel Correlations ({method})")
            plt.tight_layout()
            plt.show()
    
    # Print summary of highly correlated channels
    if side in ['right', 'both'] and results['right']['high_correlation_pairs']:
        print(f"\nHighly correlated right hand channels (|r| ≥ {threshold}):")
        for ch1, ch2, corr in results['right']['high_correlation_pairs']:
            print(f"  {ch1} ↔ {ch2}: {corr:.3f}")
    
    if side in ['left', 'both'] and results['left']['high_correlation_pairs']:
        print(f"\nHighly correlated left hand channels (|r| ≥ {threshold}):")
        for ch1, ch2, corr in results['left']['high_correlation_pairs']:
            print(f"  {ch1} ↔ {ch2}: {corr:.3f}")
    
    return results
# %%
analyze_channel_correlations(emg_data, side='both', method='pearson', threshold=0.8, plot=True)
# %%
def reduce_emg_channels(emg_data, correlation_threshold=0.8, method='pearson', 
                        reduction_method='pca'):
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
    import pandas as pd
    from sklearn.decomposition import PCA
    
    if emg_data is None or len(emg_data) == 0:
        print("No EMG data to reduce")
        return None, None
    
    # First analyze correlations
    corr_results = analyze_channel_correlations(
        emg_data, side='both', method=method, 
        threshold=correlation_threshold, plot=False
    )
    
    reduction_info = {
        'right': {'original_channels': 0, 'channel_groups': [], 'method': reduction_method},
        'left': {'original_channels': 0, 'channel_groups': [], 'method': reduction_method}
    }
    
    # Process right hand channels
    if 'right' in corr_results:
        n_channels_right = emg_data[0]['emg_right'].shape[0]
        reduction_info['right']['original_channels'] = n_channels_right
        
        # Extract channel data for correlation analysis
        right_data = np.array([[entry['emg_right'][i] for i in range(n_channels_right)] 
                              for entry in emg_data])
        right_df = pd.DataFrame(right_data, 
                               columns=[f'Right_Ch{i+1}' for i in range(n_channels_right)])
        
        # Group correlated channels
        channel_groups_right = []
        remaining_channels = set(range(1, n_channels_right + 1))
        
        # Sort correlations by strength (highest first)
        high_corr_pairs = sorted(
            corr_results['right']['high_correlation_pairs'],
            key=lambda x: abs(x[2]),
            reverse=True
        )
        
        # Group highly correlated channels
        for ch1, ch2, corr_val in high_corr_pairs:
            ch1_idx = int(ch1.replace('Right_Ch', ''))
            ch2_idx = int(ch2.replace('Right_Ch', ''))
            
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
        reduction_info['right']['channel_groups'] = [sorted(list(group)) for group in channel_groups_right]
    
    # Process left hand channels (similar to right hand)
    if 'left' in corr_results:
        n_channels_left = emg_data[0]['emg_left'].shape[0]
        reduction_info['left']['original_channels'] = n_channels_left
        
        # Extract channel data for correlation analysis
        left_data = np.array([[entry['emg_left'][i] for i in range(n_channels_left)] 
                             for entry in emg_data])
        left_df = pd.DataFrame(left_data, 
                              columns=[f'Left_Ch{i+1}' for i in range(n_channels_left)])
        
        # Group correlated channels
        channel_groups_left = []
        remaining_channels = set(range(1, n_channels_left + 1))
        
        # Sort correlations by strength (highest first)
        high_corr_pairs = sorted(
            corr_results['left']['high_correlation_pairs'],
            key=lambda x: abs(x[2]),
            reverse=True
        )
        
        # Group highly correlated channels
        for ch1, ch2, corr_val in high_corr_pairs:
            ch1_idx = int(ch1.replace('Left_Ch', ''))
            ch2_idx = int(ch2.replace('Left_Ch', ''))
            
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
        reduction_info['left']['channel_groups'] = [sorted(list(group)) for group in channel_groups_left]
    
    # Create reduced dataset by combining channels within each group
    reduced_emg_data = []
    
    for entry in emg_data:
        reduced_right = []
        reduced_left = []
        
        # Process right hand channel groups
        for group in reduction_info['right']['channel_groups']:
            # Convert to 0-based indices for array access
            indices = [ch - 1 for ch in group]
            
            if len(indices) == 1:
                # Single channel, just copy it
                reduced_right.append(entry['emg_right'][indices[0]])
            else:
                # Multiple channels to combine
                channels_to_combine = entry['emg_right'][indices]
                
                if reduction_method == 'pca':
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
                
                elif reduction_method == 'weighted_average':
                    # Calculate weights based on variance
                    variances = np.var(right_data[:, indices], axis=0)
                    weights = variances / np.sum(variances)
                    reduced_right.append(np.sum(channels_to_combine * weights))
                
                else:  # Default to simple average
                    reduced_right.append(np.mean(channels_to_combine))
        
        # Process left hand channel groups (similar to right hand)
        for group in reduction_info['left']['channel_groups']:
            # Convert to 0-based indices for array access
            indices = [ch - 1 for ch in group]
            
            if len(indices) == 1:
                # Single channel, just copy it
                reduced_left.append(entry['emg_left'][indices[0]])
            else:
                # Multiple channels to combine
                channels_to_combine = entry['emg_left'][indices]
                
                if reduction_method == 'pca':
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
                
                elif reduction_method == 'weighted_average':
                    # Calculate weights based on variance
                    variances = np.var(left_data[:, indices], axis=0)
                    weights = variances / np.sum(variances)
                    reduced_left.append(np.sum(channels_to_combine * weights))
                
                else:  # Default to simple average
                    reduced_left.append(np.mean(channels_to_combine))
        
        # Create the reduced entry
        reduced_entry = {
            'time': entry['time'],
            'emg_right': np.array(reduced_right),
            'emg_left': np.array(reduced_left)
        }
        reduced_emg_data.append(reduced_entry)
    
    # Print summary
    print("\nDimensionality Reduction Summary:")
    print(f"Reduction method: {reduction_method}")
    print(f"Right hand: {n_channels_right} → {len(reduction_info['right']['channel_groups'])} channels")
    print("Channel groups:")
    for i, group in enumerate(reduction_info['right']['channel_groups']):
        if len(group) > 1:
            print(f"  Group {i+1}: Channels {group} → Combined Channel {i+1}")
        else:
            print(f"  Group {i+1}: Channel {group[0]} → Preserved as Channel {i+1}")
    
    print(f"\nLeft hand: {n_channels_left} → {len(reduction_info['left']['channel_groups'])} channels")
    print("Channel groups:")
    for i, group in enumerate(reduction_info['left']['channel_groups']):
        if len(group) > 1:
            print(f"  Group {i+1}: Channels {group} → Combined Channel {i+1}")
        else:
            print(f"  Group {i+1}: Channel {group[0]} → Preserved as Channel {i+1}")
    
    return reduced_emg_data, reduction_info
# %%
def plot_reduced_emg_data(original_data, reduced_data, reduction_info, n_samples=500):
    """
    Plot original and reduced EMG data side by side for comparison.
    
    Args:
        original_data: Original EMG dataset
        reduced_data: Reduced EMG dataset with combined channels
        reduction_info: Dictionary with information about the reduction
        n_samples: Number of samples to plot
    """
    if original_data is None or reduced_data is None:
        print("Missing data for comparison")
        return
    
    # Extract the first n_samples measurements
    n_samples = min(n_samples, len(original_data), len(reduced_data))
    
    # Extract time data for x-axis
    time_data = np.array([entry['time'] for entry in original_data[:n_samples]])
    
    # Get the number of channels in original and reduced data
    n_orig_right = original_data[0]['emg_right'].shape[0]
    n_orig_left = original_data[0]['emg_left'].shape[0]
    n_reduced_right = reduced_data[0]['emg_right'].shape[0]
    n_reduced_left = reduced_data[0]['emg_left'].shape[0]
    
    # Create a figure with two columns: original and reduced
    fig = plt.figure(figsize=(18, max(n_orig_right + n_orig_left, n_reduced_right + n_reduced_left)))
    
    # Create separate subplots for each channel
    
    # Original right hand channels
    for i in range(n_orig_right):
        ax = plt.subplot2grid((n_orig_right + n_orig_left, 2), (i, 0))
        channel_data = np.array([entry['emg_right'][i] for entry in original_data[:n_samples]])
        ax.plot(time_data, channel_data)
        ax.set_title(f"Original Right Ch {i+1}")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        
        # Only add x-label to the bottom subplot
        if i == n_orig_right - 1:
            ax.set_xlabel("Time (s)")
    
    # Reduced right hand channels
    for i in range(n_reduced_right):
        ax = plt.subplot2grid((n_reduced_right + n_reduced_left, 2), (i, 1))
        channel_data = np.array([entry['emg_right'][i] for entry in reduced_data[:n_samples]])
        ax.plot(time_data, channel_data)
        
        # Add information about which original channels were combined
        group = reduction_info['right']['channel_groups'][i]
        if len(group) > 1:
            ax.set_title(f"Reduced Right Ch {i+1} (from original {group})")
        else:
            ax.set_title(f"Reduced Right Ch {i+1} (from original Ch {group[0]})")
        
        ax.grid(True)
        
        # Only add x-label to the bottom subplot
        if i == n_reduced_right - 1:
            ax.set_xlabel("Time (s)")
    
    # Original left hand channels
    for i in range(n_orig_left):
        ax = plt.subplot2grid((n_orig_right + n_orig_left, 2), (i + n_orig_right, 0))
        channel_data = np.array([entry['emg_left'][i] for entry in original_data[:n_samples]])
        ax.plot(time_data, channel_data)
        ax.set_title(f"Original Left Ch {i+1}")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        
        # Only add x-label to the bottom subplot
        if i == n_orig_left - 1:
            ax.set_xlabel("Time (s)")
    
    # Reduced left hand channels
    for i in range(n_reduced_left):
        ax = plt.subplot2grid((n_reduced_right + n_reduced_left, 2), (i + n_reduced_right, 1))
        channel_data = np.array([entry['emg_left'][i] for entry in reduced_data[:n_samples]])
        ax.plot(time_data, channel_data)
        
        # Add information about which original channels were combined
        group = reduction_info['left']['channel_groups'][i]
        if len(group) > 1:
            ax.set_title(f"Reduced Left Ch {i+1} (from original {group})")
        else:
            ax.set_title(f"Reduced Left Ch {i+1} (from original Ch {group[0]})")
        
        ax.grid(True)
        
        # Only add x-label to the bottom subplot
        if i == n_reduced_left - 1:
            ax.set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.suptitle(f"Comparison of Original vs. Dimensionality-Reduced EMG Data\nReduction Method: {reduction_info['right']['method']}", 
                fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.95)  # Make room for the title
    plt.show()
# %%
reduced_emg_data, reduction_info = reduce_emg_channels(emg_data, correlation_threshold=0.8, method='pearson', reduction_method='pca')
# %%
plot_reduced_emg_data(emg_data, reduced_emg_data, reduction_info, n_samples=2000)
# %%
