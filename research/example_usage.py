from data_preprocessing import DataPreprocessing

# Initialize the preprocessor
preprocessor = DataPreprocessing()

# Load and explore the data
file_path = "data/89335547/2021-06-02-1622679967-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5"
emg_data = preprocessor.read_hdf5_dataset(file_path, "/emg2qwerty/timeseries")

# Analyze correlations
correlations = preprocessor.analyze_channel_correlations(emg_data, side='both', method='pearson', threshold=0.8)

# Reduce channels
reduced_data, reduction_info = preprocessor.reduce_emg_channels(
    emg_data, 
    correlation_threshold=0.8,
    method='pearson',
    reduction_method='pca'
)

# Save the reduced data
preprocessor.save_reduced_data(reduced_data, file_path) 