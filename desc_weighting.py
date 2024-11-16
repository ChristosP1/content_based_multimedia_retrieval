import pandas as pd
from scipy.stats import wasserstein_distance
import numpy as np
import multiprocessing as mp
import os


def expand_local_descriptors(local_df):
    """
    Expands the local descriptor histograms (A3, D1, D2, D3, D4) into individual columns for each bin.
    
    Parameters:
        local_df (pd.DataFrame): DataFrame containing local descriptors with histogram columns.
        
    Returns:
        pd.DataFrame: A DataFrame with expanded local descriptor columns.
    """
    expanded_columns = []
    
    for descriptor in ['A3', 'D1', 'D2', 'D3', 'D4']:
        # Extract the list of bins and create new columns for each bin
        bins = local_df[descriptor].apply(eval)  # Assuming these are stored as strings that need to be evaluated to lists
        bins_df = pd.DataFrame(bins.tolist(), columns=[f'{descriptor}_{i+1}' for i in range(len(bins[0]))])
        expanded_columns.append(bins_df)

    # Concatenate the expanded columns into a single dataframe
    expanded_df = pd.concat(expanded_columns, axis=1)

    # Return expanded local descriptor dataframe
    return expanded_df


def merge_global_and_local_descriptors(global_desc_file, local_desc_file, output_file):
    """
    Merges global and expanded local descriptors into a single dataframe and saves the result as a CSV file.
    
    Parameters:
        global_desc_file (str): Path to the global descriptors CSV file.
        local_desc_file (str): Path to the local descriptors CSV file.
        output_file (str): Path to the output CSV file for saving the merged descriptors.
    """
    # Load the global and local descriptors
    global_descriptors_df = pd.read_csv(global_desc_file)
    local_descriptors_df = pd.read_csv(local_desc_file)

    # Debug: Print shapes of both dataframes before merging
    # print(f"Global Descriptors Shape: {global_descriptors_df.shape}")
    # print(f"Local Descriptors Shape: {local_descriptors_df.shape}")
    
    # Find the common file names between global and local descriptors
    common_files = set(global_descriptors_df['file_name']).intersection(set(local_descriptors_df['file_name']))

    # Filter global and local descriptors to only include rows with common file names
    global_descriptors_df = global_descriptors_df[global_descriptors_df['file_name'].isin(common_files)].sort_values(by=['file_name']).reset_index(drop=True)
    local_descriptors_df = local_descriptors_df[local_descriptors_df['file_name'].isin(common_files)].sort_values(by=['file_name']).reset_index(drop=True)

    # Drop the obj_class and file_name from global descriptors (we get them from local descriptors)
    global_descriptors_df = global_descriptors_df.drop(columns=['obj_class', 'file_name'])

    # Expand the local descriptor histograms into individual columns
    expanded_local_descriptors_df = expand_local_descriptors(local_descriptors_df)

    # Extract file_name and obj_class from the local descriptors
    file_and_class_df = local_descriptors_df[['file_name', 'obj_class']]

    # Concatenate file_name + obj_class, global descriptors, and expanded local descriptors
    merged_df = pd.concat([file_and_class_df, global_descriptors_df, expanded_local_descriptors_df], axis=1).sort_values(by=['obj_class']).reset_index(drop=True)

    # Debug: Print shapes after merging
    # print(f"Merged Descriptors Shape: {merged_df.shape}")
    
    # Save the merged dataframe to a CSV file
    merged_df.to_csv(output_file, index=False)
    # print(f"Descriptors merged and saved to {output_file}")
    

def compute_pairwise_distances_single_value(df, feature_name):
    """
    Compute pairwise distances for single-value features and return the distance matrix, mean, and std deviation.
    """
    distance_matrix = []
    
    for i in range(len(df)):
        distances = []
        for j in range(len(df)):
            if i != j:
                dist = abs(df[feature_name].values[i] - df[feature_name].values[j])
                distances.append(dist)
            else:
                distances.append(0)
        distance_matrix.append(distances)
    
    distance_matrix = np.array(distance_matrix)
    
    # Compute mean and standard deviation for the distances
    mean_dist = np.mean(distance_matrix)
    std_dist = np.std(distance_matrix)
    
    # Z-score standardize the distance matrix
    standardized_distances = (distance_matrix - mean_dist) / std_dist
    
    return standardized_distances, mean_dist, std_dist


def save_standardization_params(feature, mean, std, output_path):
    """
    Save the standardization (mean and std) for each feature to a CSV file.
    """
    # Create a new entry for the feature
    new_entry = pd.DataFrame({"descriptor": [feature], "mean": [mean], "std": [std]})
    
    # Append it to the CSV file
    new_entry.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    # print(f"Saved {feature} to {output_path}")
    

def compute_pairwise_distances_histogram(df, feature_prefix):
    """
    Compute pairwise EMD distances for histogram-based features and return the mean and std deviation.
    """
    distance_matrix = []

    # Loop through all pairs of objects to compute pairwise distances
    for i in range(len(df)):
        distances = []
        for j in range(len(df)):
            if i != j:
                # Extract histogram values for feature_prefix
                hist_i = df.loc[i, [col for col in df.columns if col.startswith(f'{feature_prefix}_')]].values
                hist_j = df.loc[j, [col for col in df.columns if col.startswith(f'{feature_prefix}_')]].values

                # Compute Wasserstein distance (EMD) between histograms
                emd_total = wasserstein_distance(hist_i, hist_j)
                distances.append(emd_total)
            else:
                distances.append(0)  # Distance to itself is 0
        distance_matrix.append(distances)

    # Convert distance_matrix to numpy array for easy manipulation
    distance_matrix = np.array(distance_matrix)
    
    # Compute mean and standard deviation of the distances
    mean_dist = np.mean(distance_matrix)
    std_dist = np.std(distance_matrix)
    
    return mean_dist, std_dist


def compute_histogram_distance_range_parallel(feature, df):
    """
    Wrapper function to compute mean and std for histogram-based distances.
    This will be executed in parallel.
    """
    mean_dist, std_dist = compute_pairwise_distances_histogram(df, feature)
    return feature, mean_dist, std_dist


def parallel_compute_histogram_ranges(df, histogram_features):
    """
    Parallelized function to compute distance ranges (mean and std) for all histogram features.
    """
    # Create a pool of processes
    with mp.Pool(processes=len(histogram_features)) as pool:
        # Map the compute function to the histogram features
        results = pool.starmap(compute_histogram_distance_range_parallel, [(feature, df) for feature in histogram_features])
    
    # Convert results to a dictionary where keys are features and values are (mean, std)
    distance_ranges = {feature: (np.round(mean_dist,4), np.round(std_dist, 4)) for feature, mean_dist, std_dist in results}
    
    return distance_ranges
    

def prepare_distance_weighting_file(file_path):
    """
    Removes the existing distance_weighting_params.csv file if it exists and creates a new one.
    """
    # Check if the file exists
    if os.path.exists(file_path):
        # Remove the existing file
        os.remove(file_path)
        print(f"File {file_path} removed.")

    # Create a new empty file with headers
    params_df = pd.DataFrame(columns=['descriptor', 'mean', 'std'])
    params_df.to_csv(file_path, index=False)
    print(f"New file created at {file_path}.")
    

if __name__=="__main__":
    
    global_desc_file = 'outputs/data/global_descriptors_standardized.csv'
    local_desc_file = 'outputs/data/local_descriptors.csv'
    combined_desc_path = 'outputs/data/combined_descriptors.csv' 
    bins_path = 'outputs/data/average_bins_local.csv'

    # Merge descriptor files
    merge_global_and_local_descriptors(
        global_desc_file,
        local_desc_file,
        combined_desc_path
    )
    
    # Read csv files
    combined_desc_df = pd.read_csv(combined_desc_path)
    bins_df = pd.read_csv(bins_path)
    prepare_distance_weighting_file('outputs/data/distance_weighting_params.csv')
    distance_weighting_params_path = 'outputs/data/distance_weighting_params.csv'
    
    # ======================== Distance standardization params for Global Descriptors ======================= #
    single_value_features = ["volume", "surface_area", "diameter", "eccentricity", "compactness", "rectangularity", "convexity", "sphericity", "elongation"]

    for feature in single_value_features:
        std_distances, mean_dist, std_dist = compute_pairwise_distances_single_value(combined_desc_df, feature)
        # print(f"{feature}: Mean = {mean_dist}, Std = {std_dist}")
        save_standardization_params(feature, np.round(mean_dist, 4), np.round(std_dist, 4), distance_weighting_params_path)

    
    # ======================== Distance standardization params for Local Descriptors ======================== #
    histogram_features = ["A3", "D1", "D2", "D3", "D4"]

    # Compute distance ranges in parallel for local features
    distance_ranges = parallel_compute_histogram_ranges(combined_desc_df, histogram_features)
    
    # Save the mean and std for histogram distances
    for feature, (mean_dist, std_dist) in distance_ranges.items():
        save_standardization_params(feature, mean_dist, std_dist, distance_weighting_params_path)