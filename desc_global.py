import numpy as np
import trimesh
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')
import logging
import os
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
import time
import multiprocessing as mp


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


def compute_voxel_volume(mesh, pitch=0.01):
    """
    Approximates the volume of a non-watertight mesh using voxelization.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        pitch (float): The size of the voxels.
        
    Returns:
        float: The estimated volume of the mesh based on voxelization.
    """
    # Voxelize the mesh
    voxelized = mesh.voxelized(pitch)
    
    # The total volume is the number of filled voxels times the volume of each voxel
    total_volume = voxelized.volume
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
    # volume = mesh.volume
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
    # volume = mesh.volume
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


def compute_convexity(mesh, volume):
    """
    Computes the convexity of the mesh.
    Convexity is defined as the ratio of the mesh volume to the volume of its convex hull.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        
    Returns:
        float: The convexity of the mesh.
        
    Raises:
        ValueError: If the mesh is not watertight (no volume).
    """
    convex_hull = mesh.convex_hull
    convex_hull_volume = compute_voxel_volume(convex_hull)
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


def compute_descriptors(preprocessed_mesh, original_mesh, obj_class):
    """
    Compute global descriptors for the preprocessed mesh.
    
    Parameters:
        preprocessed_mesh (trimesh.Trimesh): The input mesh after preprocessing.
        original_mesh (trimesh.Trimesh, optional): The original unprocessed mesh.
        obj_class (str, optional): The class or label of the object.
        
    Returns:
        dict: A dictionary of global descriptors including volume, surface area, 
              diameter, eccentricity, compactness, rectangularity, convexity, 
              sphericity, and elongation.
    """
    
    features = {}
    logger.info(obj_class)
    # Compute surface area, diameter, and eccentricity from the preprocessed mesh
    if preprocessed_mesh:
        volume = compute_voxel_volume(preprocessed_mesh, pitch=0.01)
        features['volume'] = volume
        features['surface_area'] = compute_surface_area(preprocessed_mesh)
        features['diameter'] = get_diameter(preprocessed_mesh)
        features['eccentricity'] = compute_eccentricity(preprocessed_mesh)
        compactness = compute_compactness(preprocessed_mesh, volume)
        features['compactness'] = compactness
        features['rectangularity'] = compute_rectangularity(preprocessed_mesh, volume)
        features['convexity'] = compute_convexity(preprocessed_mesh, volume)
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
        

def standardize_column_z_score(column, mean=None, std=None):
    """Standardize a single column."""
    if mean is None or std is None:
        mean = np.mean(column)
        std = np.std(column)
    
    standardized_column = (column - mean) / std
    return standardized_column, mean, std


def standardize_features(global_descriptors_df):
    
    # List of features to standardize
    features_to_standardize = ["surface_area", "diameter", "eccentricity", 
                               "compactness", "rectangularity", "convexity", "sphericity", "elongation"]
    
    # Dictionary to save standardization parameters
    standardization_dict = {"feature": [], "mean": [], "std": []}
    
    for feature in features_to_standardize:
        # Standardize each feature's column
        global_descriptors_df[feature], mean, std = standardize_column_z_score(global_descriptors_df[feature])
        
        # Save the standardization parameters (mean and std)
        standardization_dict["feature"].append(feature)
        standardization_dict["mean"].append(mean)
        standardization_dict["std"].append(std)
    
    # Save the standardized data
    global_descriptors_df.to_csv('outputs/data/standardized_global_descriptors.csv', index=False)
    
    # Save the standardization parameters (if needed for later use)
    standardization_params_df = pd.DataFrame(standardization_dict)
    standardization_params_df.to_csv('outputs/data/standardization_params.csv', index=False)
    


def plot_correlations(global_descriptors_df):
    # Step 1: Exclude non-numeric columns (such as 'obj_class' and 'file_name')
    numeric_columns = global_descriptors_df.select_dtypes(include=[float, int]).columns
    
    # Step 2: Group by 'obj_class' and calculate the variance for only numeric columns
    grouped_variance_df = global_descriptors_df.groupby('obj_class')[numeric_columns].var()
    
    # Step 3: Plot a heatmap of variance for each descriptor per class
    plt.figure(figsize=(12, 20))
    sns.heatmap(grouped_variance_df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Variance'}, fmt=".2f")
    plt.title('Variance of Descriptors for Each Class (Lower is Better)')
    plt.xlabel('Descriptor')
    plt.ylabel('Class')
    
    # Save the plot using the full path
    plt.savefig('outputs/plots/global_descriptors.png')
    
    # Optional: close the plot to free memory
    plt.close()
    

def detect_outliers_iqr(column):
    """
    Detect outliers using the IQR (Interquartile Range) method.
    :param column: A Pandas Series representing a column of data
    :return: A boolean mask indicating whether each value is an outlier
    """
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (column < lower_bound) | (column > upper_bound)


def find_outliers_iqr(data, category_column, feature_columns, iqr_multiplier=3.0):
    """
    Detect outliers within each category using the IQR method.
    
    Parameters:
        data (pd.DataFrame): The dataset containing objects with features and categories.
        category_column (str): The name of the column containing object categories.
        feature_columns (list): The list of feature columns to check for outliers.
        
    Returns:
        pd.DataFrame: A DataFrame containing only the objects that are not outliers.
    """
    filtered_data = data.copy()
    
    for category in data[category_column].unique():
        category_data = data[data[category_column] == category]
        
        for feature in feature_columns:
            q1 = category_data[feature].quantile(0.25)
            q3 = category_data[feature].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            # Remove objects outside of the lower and upper bounds
            filtered_data = filtered_data[
                ~(filtered_data[category_column] == category) | 
                ((category_data[feature] >= lower_bound) & (category_data[feature] <= upper_bound))
            ]
    
    return filtered_data



def compute_descriptors(preprocessed_mesh, original_mesh, obj_class):
    """
    Compute global descriptors for the preprocessed mesh.
    
    Parameters:
        preprocessed_mesh (trimesh.Trimesh): The input mesh after preprocessing.
        original_mesh (trimesh.Trimesh, optional): The original unprocessed mesh.
        obj_class (str, optional): The class or label of the object.
        
    Returns:
        dict: A dictionary of global descriptors.
    """
    features = {}
    logger.info(obj_class)
    # Compute surface area, diameter, and eccentricity from the preprocessed mesh
    if preprocessed_mesh:
        volume = compute_voxel_volume(preprocessed_mesh, pitch=0.01)
        features['volume'] = volume
        features['surface_area'] = compute_surface_area(preprocessed_mesh)
        features['diameter'] = get_diameter(preprocessed_mesh)
        features['eccentricity'] = compute_eccentricity(preprocessed_mesh)
        compactness = compute_compactness(preprocessed_mesh, volume)
        features['compactness'] = compactness
        features['rectangularity'] = compute_rectangularity(preprocessed_mesh, volume)
        features['convexity'] = compute_convexity(preprocessed_mesh, volume)
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


def compute_global_descriptors_parallel(df_chunk, original_dataset_path, preprocessed_dataset_path):
    """
    Computes global descriptors in parallel for each chunk of the DataFrame.
    
    Parameters:
        df_chunk (pd.DataFrame): A chunk of the DataFrame containing shapes metadata.
        original_dataset_path (str): The path to the original dataset.
        preprocessed_dataset_path (str): The path to the preprocessed dataset.
    
    Returns:
        list: A list of dictionaries containing global descriptors for each shape.
    """
    global_descriptors = []

    for _, row in df_chunk.iterrows():
        obj_class = row['obj_class']
        file_name = row['file_name']
        
        # Get file paths
        original_file_path = os.path.join(original_dataset_path, obj_class, file_name)
        preprocessed_file_path = os.path.join(preprocessed_dataset_path, obj_class, file_name)
        
        # Load original and preprocessed meshes
        original_mesh = load_mesh(original_file_path)
        preprocessed_mesh = load_mesh(preprocessed_file_path)
        
        # Compute global descriptors
        descriptors = compute_descriptors(preprocessed_mesh, original_mesh, obj_class)
        
        # Add object class and file name for reference
        descriptors['obj_class'] = obj_class
        descriptors['file_name'] = file_name

        global_descriptors.append(descriptors)

    return global_descriptors


def run_global_descriptors_parallel(preprocessed_shapes_df, original_dataset_path, preprocessed_dataset_path, num_processes=4):
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
                               [(df_chunk, original_dataset_path, preprocessed_dataset_path) for df_chunk in df_chunks])

    # Flatten the list of results (since each process returns a list of descriptors)
    flattened_results = [item for sublist in results for item in sublist]
    
    # Convert the list of descriptors into a DataFrame
    global_descriptors_df = pd.DataFrame(flattened_results)
    
    return global_descriptors_df



def main():
    """Main function to load data, compute descriptors, and save results."""
    
    ORIGINAL_DATASET_PATH = 'datasets/dataset_original'
    PREPROCESSED_DATASET_PATH = 'datasets/dataset_snippet_medium_normalized'
    OUTPUTS_DATA_PATH = 'outputs/data'
    
    original_dataset_csv_path = 'outputs/shapes_data.csv'
    preprocessed_dataset_csv_path = 'outputs/shapes_data_normalized.csv'
    times_path = os.path.join(OUTPUTS_DATA_PATH, "times.csv")
    
    # Load original and preprocessed shape metadata
    original_shapes_df = pd.read_csv(original_dataset_csv_path)
    preprocessed_shapes_df = pd.read_csv(preprocessed_dataset_csv_path)

    times_df = pd.read_csv(os.path.join(OUTPUTS_DATA_PATH, "times.csv"))
    
    start_global_descriptors = time.time()
        
    num_processes = 10
    
    global_descriptors_df = run_global_descriptors_parallel(preprocessed_shapes_df, 
                                                            ORIGINAL_DATASET_PATH, 
                                                            PREPROCESSED_DATASET_PATH, 
                                                            num_processes=num_processes)
    
    
    
    
    
    # Apply IQR filtering to remove outliers within each category
    feature_columns = ['volume', 'surface_area', 'diameter', 'eccentricity', 
                       'compactness', 'rectangularity', 'convexity', 'sphericity', 'elongation']

    # global_descriptors_df = find_outliers_iqr(global_descriptors_df, 'obj_class', feature_columns)
    
    
    # Save the final descriptors to a CSV file
    global_descriptors_df.to_csv(os.path.join(OUTPUTS_DATA_PATH, 'global_descriptors_non_standardized.csv'), index=False)
    
    # Apply z-score standardization to the cleaned DataFrame
    standardize_features(global_descriptors_df)

    # Print the descriptors for verification
    print(global_descriptors_df.head())
    
    plot_correlations(global_descriptors_df)
    
    end_global_descriptors = time.time()
    
    times_df['global_desc'] = end_global_descriptors - start_global_descriptors
    times_df.to_csv(times_path, index=False)


if __name__ == "__main__":
    main()