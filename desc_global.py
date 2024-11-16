import numpy as np
import trimesh
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')
import logging
import random
import time
import multiprocessing as mp
import os
import pickle
from sklearn.ensemble import IsolationForest


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()


def compute_surface_area(mesh):
    """
    Computes the surface area of the mesh.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        
    Returns:
        float: The surface area of the mesh.
    """
    
    return mesh.area


def compute_voxel_volume(mesh, file_path, pitch=0.01):
    """
    Approximates the volume of a non-watertight mesh using voxelization.
    Saves and retrieves the voxelized mesh from disk.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        file_path (str): Path to the saved voxelized volume.
        pitch (float): The size of the voxels.
        
    Returns:
        float: The estimated volume of the mesh based on voxelization.
    """
    # Check if the voxelized volume already exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            voxelized_volume = pickle.load(f)
        return voxelized_volume
    
    # Voxelize the mesh if not already saved
    voxelized = mesh.voxelized(pitch)
    total_volume = voxelized.volume
    
    # Save the voxelized volume
    with open(file_path, 'wb') as f:
        pickle.dump(total_volume, f)
    
    return total_volume


def compute_compactness(mesh, volume):
    """
    Computes the compactness (sphericity) of the mesh.
    The compactness equals one if the shape is a perfect sphere.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        
    Returns:
        float: The compactness of the mesh.
        
    Raises:
        ValueError: If the mesh is not watertight (no volume).
    """
    
    area = compute_surface_area(mesh)
    compactness = (area ** 3) / (36 * np.pi * (volume ** 2))
    return compactness


def compute_rectangularity(mesh, volume):
    """
    Computes the 3D rectangularity of the mesh.
    Rectangularity is defined as the ratio of the mesh volume to the volume of its oriented bounding box (OBB).
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        
    Returns:
        float: The rectangularity of the mesh.
        
    Raises:
        ValueError: If the mesh is not watertight (no volume).
    """
    obb = mesh.bounding_box_oriented        # Get the oriented bounding box (OBB)
    rectangularity = min((volume / obb.volume), 1)    # OBB SO IT IS ORIENTATION INDEPENDENT !!!!!!!!!!!!!!!!!!!!
    rectangularity = max(rectangularity, 0)
    
    return rectangularity


def get_diameter(mesh, method="fast"):
    '''given a mesh, get the furthest points on the convex haul and then try all possible combinations
    of the distances between points and return the max one'''

    convex_hull = mesh.convex_hull
    max_dist = 0
    vertices = list(convex_hull.vertices)
    
    if method == "fast": # if fast method, REDUCE nr vertices
        """SAMPLE 200 VERTICES"""
    
        if len(vertices) > 200:
            vertices = random.sample(vertices, 200)
    
    if method == "slow": # do nothing, just calculate between all pairs
        pass
    
    # find maximum distance between two vertices
    for i in range(len(vertices)):
        for j in range(i, len(vertices)):
            dist = np.linalg.norm(vertices[i] - vertices[j])
            if dist > max_dist:
                max_dist = dist
    
    return max_dist


def compute_convexity(mesh, volume, file_name):
    """
    Computes the convexity of the mesh.
    Convexity is defined as the ratio of the mesh volume to the volume of its convex hull.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        volume (float): The volume of the mesh.
        file_name (str): The file name of the mesh for saving voxelized volumes.
        
    Returns:
        float: The convexity of the mesh.
    """
    
    # Define file path for the voxelized convex hull volume
    voxel_folder = 'voxelized_volumes'
    os.makedirs(voxel_folder, exist_ok=True)
    convex_hull_file_path = os.path.join(voxel_folder, f'{file_name}_convex_hull_voxel_volume.pkl')
    
    # Compute the convex hull volume
    convex_hull = mesh.convex_hull
    convex_hull_volume = compute_voxel_volume(convex_hull, convex_hull_file_path)
    
    # Compute convexity
    convexity = min((volume / convex_hull_volume), 1.0)
    convexity = max(0, convexity)
    return convexity


def compute_eccentricity(mesh):
    """
    Computes the eccentricity of the mesh.
    Eccentricity is defined as the ratio of the largest to smallest eigenvalues of the covariance matrix of the vertices.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        
    Returns:
        float: The eccentricity of the mesh.
    """
    eigenvalues, _ = get_eigen(mesh)
    # Sort the eigenvalues in ascending order
    eigenvalues = np.sort(np.abs(eigenvalues))
    
    lambda_3  = eigenvalues[0]
    lambda_1  = eigenvalues[-1]
    
    # Handle zero eigenvalues to prevent division by zero
    if lambda_3 == 0:
        lambda_3 = sys.float_info.min  # Smallest positive float
    
    eccentricity = lambda_1 / lambda_3
    return eccentricity


def get_eigen(mesh):
    """
    Computes the eigenvalues and eigenvectors of the covariance matrix of the mesh vertices.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        
    Returns:
        tuple: (eigenvalues, eigenvectors)
    """
    vertices = mesh.vertices
    # Compute covariance matrix; set rowvar=False because each row is an observation (vertex)
    covariance_matrix = np.cov(vertices, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return eigenvalues, eigenvectors


def compute_sphericity(compactness):
    """
    Computes the sphericity of a mesh.
    
    Parameters:
        mesh (trimesh.Trimesh): The input 3D mesh.
        
    Returns:
        dict: Sphericity value.
    """
    
    sphericity = min((1/compactness), 1.0)
    sphericity = max(0, sphericity)

    return  sphericity


def compute_elongation(mesh):
    """
    Compute the elongation of the mesh based on the bounding box.
    
    Parameters:
        mesh (trimesh.Trimesh): The input 3D mesh.
        
    Returns:
        dict: Elongation value of the bounding box.
    """
    # Get the extents of the bounding box
    extents = mesh.bounding_box_oriented.extents
    sorted_extents = np.sort(extents)  # Sort the extents to find longest and second longest

    elongation = sorted_extents[-1] / sorted_extents[-2]

    return elongation


def load_mesh(file_path):
    """Load a 3D mesh from a given file path."""
    try:
        return trimesh.load(file_path)
    except Exception as e:
        logger.error(f"Could not load mesh: {file_path}, error: {e}")
        return None


def compute_descriptors(preprocessed_mesh, file_name):
    """
    Compute global descriptors for the preprocessed mesh.
    
    Parameters:
        preprocessed_mesh (trimesh.Trimesh): The input mesh after preprocessing.
        original_mesh (trimesh.Trimesh, optional): The original unprocessed mesh.
        obj_class (str, optional): The class or label of the object.
        file_name (str): The file name of the mesh for saving voxelized volumes.
        
    Returns:
        dict: A dictionary of global descriptors including volume, surface area, 
              diameter, eccentricity, compactness, rectangularity, convexity, 
              sphericity, and elongation.
    """
    
    # Define file path for the voxelized volume
    voxel_folder = 'voxelized_volumes'
    os.makedirs(voxel_folder, exist_ok=True)
    voxel_file_path = os.path.join(voxel_folder, f'{file_name}_voxel_volume.pkl')
    
    features = {}
    
    # Compute surface area, diameter, and eccentricity from the preprocessed mesh
    if preprocessed_mesh:
        volume = compute_voxel_volume(preprocessed_mesh, voxel_file_path, pitch=0.01)
        
        features['volume'] = volume
        features['surface_area'] = compute_surface_area(preprocessed_mesh)
        features['diameter'] = get_diameter(preprocessed_mesh)
        features['eccentricity'] = compute_eccentricity(preprocessed_mesh)
        compactness = compute_compactness(preprocessed_mesh, volume)
        features['compactness'] = compactness
        features['rectangularity'] = compute_rectangularity(preprocessed_mesh, volume)
        features['convexity'] = compute_convexity(preprocessed_mesh, volume, file_name)
        features['sphericity'] = compute_sphericity(compactness)
        features['elongation'] = compute_elongation(preprocessed_mesh)
    else:
        features['surface_area'] = np.nan
        features['diameter'] = np.nan
        features['eccentricity'] = np.nan
        features['compactness'] = np.nan
        features['rectangularity'] = np.nan
        features['convexity'] = np.nan
        features['sphericity'] = np.nan
        features['elongation'] = np.nan
    
    return features


def min_max_standardize_column(column):
    """
    Min-max standardize a column to the range [0, 1].
    
    Parameters:
        column (pd.Series): The column to standardize.
    
    Returns:
        pd.Series: The standardized column.
        float: The minimum value (for future reference).
        float: The maximum value (for future reference).
    """
    col_min = column.min()
    col_max = column.max()
    
    # Apply min-max standardization
    standardized_column = (column - col_min) / (col_max - col_min) if col_max > col_min else column
    
    return standardized_column, col_min, col_max


def z_score_standardize_and_save_params(global_descriptors_df):
    """
    Standardize global descriptors using z-score normalization and save the parameters.
    
    Parameters:
        global_descriptors_df (pd.DataFrame): DataFrame containing the global descriptors.
        output_path (str): Path to save the normalization parameters (mean and std) to a CSV file.
        
    Returns:
        pd.DataFrame: The standardized global descriptors DataFrame.
    """
    # List of features to standardize
    features_to_standardize = ["volume", "surface_area", "diameter", "eccentricity", 
                               "compactness", "rectangularity", "convexity", "sphericity", "elongation"]
    
    # Dictionary to store mean and standard deviation for each feature
    normalization_params = {"feature": [], "mean": [], "std": []}
    
    # Standardize each feature
    for feature in features_to_standardize:
        # Compute mean and standard deviation
        col_mean = global_descriptors_df[feature].mean()
        col_std = global_descriptors_df[feature].std()
        
        # Save the parameters
        normalization_params["feature"].append(feature)
        normalization_params["mean"].append(col_mean)
        normalization_params["std"].append(col_std)
        
        # Apply z-score normalization
        global_descriptors_df[feature] = (global_descriptors_df[feature] - col_mean) / col_std if col_std > 0 else global_descriptors_df[feature]
    
    # Save normalization parameters to a CSV file
    normalization_params_df = pd.DataFrame(normalization_params)
    normalization_params_df.to_csv('outputs/data/standardization_params_z_score.csv', index=False)
    
    return global_descriptors_df




def apply_isolation_forest(df, feature_columns, contamination=0.03):
    """
    Apply Isolation Forest separately to each feature column and remove rows
    where any column is identified as an outlier.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the features.
        feature_columns (list): The list of feature columns to apply the Isolation Forest on.
        contamination (float): The expected proportion of outliers in the data.
        
    Returns:
        pd.DataFrame: A cleaned DataFrame with outliers removed.
    """
    # Initialize an empty mask that will mark all non-outliers across all features
    mask = pd.Series([True] * len(df), index=df.index)
    
    for feature in feature_columns:
        # Apply Isolation Forest to the current feature column
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        
        # Reshape the feature to match the format required by IsolationForest (2D array)
        feature_data = df[feature].values.reshape(-1, 1)
        
        # Predict inliers (1) and outliers (-1)
        outlier_predictions = iso_forest.fit_predict(feature_data)
        
        # Update mask: keep only the rows where the current feature is NOT an outlier
        mask = mask & (outlier_predictions == 1)
    
    # Filter the DataFrame using the mask (only keep rows where all features are non-outliers)
    cleaned_df = df[mask]
    
    print(f"Number of outliers removed: {len(df) - len(cleaned_df)}")
    
    return cleaned_df


def remove_outliers_two_std(df, feature_columns, threshold=2):
    """
    Remove rows where any feature column has a value that is more than two standard deviations away from the mean.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the features.
        feature_columns (list): The list of feature columns to apply the two std rule.
        threshold (int, optional): The number of standard deviations from the mean to be considered as outlier.
        
    Returns:
        pd.DataFrame: A cleaned DataFrame with outliers removed.
    """
    # Initialize an empty mask that will mark all rows as inliers
    mask = pd.Series([True] * len(df), index=df.index)
    
    for feature in feature_columns:
        # Calculate the mean and standard deviation for the current feature
        mean = df[feature].mean()
        std = df[feature].std()

        # Create a mask that keeps only rows where the value is within `threshold` std from the mean
        feature_mask = (df[feature] >= (mean - threshold * std)) & (df[feature] <= (mean + threshold * std))
        
        # Update the overall mask to keep rows where all features are within the threshold
        mask = mask & feature_mask
    
    # Filter the DataFrame using the mask (only keep rows where all features are within the threshold)
    cleaned_df = df[mask]
    
    print(f"Number of outliers removed: {len(df) - len(cleaned_df)}")
    
    return cleaned_df


def compute_global_descriptors_parallel(df_chunk, preprocessed_dataset_path):
    """
    Computes global descriptors in parallel for each chunk of the DataFrame.
    
    Parameters:
        df_chunk (pd.DataFrame): A chunk of the DataFrame containing shapes metadata.
        preprocessed_dataset_path (str): The path to the preprocessed dataset.
    
    Returns:
        list: A list of dictionaries containing global descriptors for each shape.
    """
    global_descriptors = []

    for _, row in df_chunk.iterrows():
        obj_class = row['obj_class']
        file_name = row['file_name']
        
        # Get file paths
        preprocessed_file_path = os.path.join(preprocessed_dataset_path, obj_class, file_name)
        
        # Load original and preprocessed meshes
        preprocessed_mesh = load_mesh(preprocessed_file_path)
        
        # Compute global descriptors
        descriptors = compute_descriptors(preprocessed_mesh, file_name)
        
        # Add object class and file name for reference
        descriptors['obj_class'] = obj_class
        descriptors['file_name'] = file_name

        global_descriptors.append(descriptors)

    return global_descriptors


def run_global_descriptors_parallel(preprocessed_shapes_df, preprocessed_dataset_path, num_processes=4):
    """
    Run the global descriptor computations in parallel using multiprocessing.
    
    Parameters:
        preprocessed_shapes_df (pd.DataFrame): The dataframe containing the preprocessed shapes.
        original_dataset_path (str): Path to the original dataset.
        preprocessed_dataset_path (str): Path to the preprocessed dataset.
        num_processes (int): Number of parallel processes to run.
        
    Returns:
        pd.DataFrame: A DataFrame containing all the computed global descriptors.
    """
    # Split the dataframe into chunks, one for each process
    df_chunks = np.array_split(preprocessed_shapes_df, num_processes)

    # Use multiprocessing to compute descriptors in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(compute_global_descriptors_parallel, 
                               [(df_chunk, preprocessed_dataset_path) for df_chunk in df_chunks])

    # Flatten the list of results (since each process returns a list of descriptors)
    flattened_results = [item for sublist in results for item in sublist]
    
    # Convert the list of descriptors into a DataFrame
    global_descriptors_df = pd.DataFrame(flattened_results)
    
    return global_descriptors_df



def main():
    """Main function to load data, compute descriptors, and save results."""
    
    PREPROCESSED_DATASET_PATH = 'datasets/dataset_original_normalized'
    OUTPUTS_DATA_PATH = 'outputs/data'
    
    preprocessed_dataset_csv_path = 'outputs/shapes_data_normalized.csv'
    times_path = os.path.join(OUTPUTS_DATA_PATH, "times.csv")
    
    # Load original and preprocessed shape metadata
    preprocessed_shapes_df = pd.read_csv(preprocessed_dataset_csv_path)

    times_df = pd.read_csv(os.path.join(OUTPUTS_DATA_PATH, "times.csv"))
    
    start_global_descriptors = time.time()
        
    num_processes = 10
    
    global_descriptors_df = run_global_descriptors_parallel(preprocessed_shapes_df, 
                                                            PREPROCESSED_DATASET_PATH, 
                                                            num_processes=num_processes)
    
    
    # List of features to check for outliers
    feature_columns = ["volume", "surface_area", "diameter", "eccentricity", 
                       "compactness", "rectangularity", "convexity", "sphericity", "elongation"]
    
    # Apply Isolation Forest to remove outliers
    # cleaned_df = apply_isolation_forest(global_descriptors_df, feature_columns, contamination=0.01)
    
    cleaned_df = remove_outliers_two_std(global_descriptors_df, feature_columns, threshold=3)
    
    # Save the final descriptors to a CSV file
    cleaned_df.to_csv(os.path.join(OUTPUTS_DATA_PATH, 'global_descriptors_non_standardized.csv'), index=False)
    
    # Apply min-max standardization to the cleaned DataFrame
    cleaned_df = z_score_standardize_and_save_params(cleaned_df)
    
    cleaned_df = cleaned_df.round(4)
    
    # Save the final descriptors to a CSV file
    cleaned_df.to_csv(os.path.join(OUTPUTS_DATA_PATH, 'global_descriptors_standardized.csv'), index=False)

    # Print the descriptors for verification
    print(cleaned_df.head())
        
    end_global_descriptors = time.time()
    
    times_df['global_desc'] = end_global_descriptors - start_global_descriptors
    times_df.to_csv(times_path, index=False)


if __name__ == "__main__":
    main()