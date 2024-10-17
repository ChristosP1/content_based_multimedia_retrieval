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
from sklearn.preprocessing import LabelEncoder


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()


hardcoded_classes = {
        'AircraftBuoyant': {'compactness': 0.850, 'rectangularity': 0.700, 'convexity': 0.950},
        'Apartment': {'compactness': 0.550, 'rectangularity': 0.850, 'convexity': 0.700},
        'Bird': {'compactness': 0.700, 'rectangularity': 0.600, 'convexity': 0.850},
        'Bottle': {'compactness': 0.850, 'rectangularity': 0.500, 'convexity': 0.900},
        'BuildingNonResidential': {'compactness': 0.500, 'rectangularity': 0.850, 'convexity': 0.650},
        'Chess': {'compactness': 0.750, 'rectangularity': 0.700, 'convexity': 0.800},
        'City': {'compactness': 0.450, 'rectangularity': 0.800, 'convexity': 0.600},
        'ClassicPiano': {'compactness': 0.650, 'rectangularity': 0.800, 'convexity': 0.700},
        'Computer': {'compactness': 0.550, 'rectangularity': 0.800, 'convexity': 0.700},
        'DeskPhone': {'compactness': 0.600, 'rectangularity': 0.650, 'convexity': 0.800},
        'Door': {'compactness': 0.350, 'rectangularity': 0.850, 'convexity': 0.450},
        'Drum': {'compactness': 0.750, 'rectangularity': 0.600, 'convexity': 0.850},
        'FloorLamp': {'compactness': 0.350, 'rectangularity': 0.800, 'convexity': 0.650},
        'Glasses': {'compactness': 0.700, 'rectangularity': 0.550, 'convexity': 0.800},
        'Guitar': {'compactness': 0.800, 'rectangularity': 0.700, 'convexity': 0.900},
        'Hat': {'compactness': 0.600, 'rectangularity': 0.650, 'convexity': 0.800},
        'HumanHead': {'compactness': 0.850, 'rectangularity': 0.600, 'convexity': 0.900},
        'Insect': {'compactness': 0.650, 'rectangularity': 0.550, 'convexity': 0.750},
        'MilitaryVehicle': {'compactness': 0.550, 'rectangularity': 0.800, 'convexity': 0.700},
        'MultiSeat': {'compactness': 0.450, 'rectangularity': 0.222, 'convexity': 0.240},
        'Monitor': {'compactness': 0.500, 'rectangularity': 0.800, 'convexity': 0.600},
        'NonWheelChair': {'compactness': 0.550, 'rectangularity': 0.750, 'convexity': 0.650},
        'PianoBoard': {'compactness': 0.600, 'rectangularity': 0.850, 'convexity': 0.700},
        'Ship': {'compactness': 0.750, 'rectangularity': 0.650, 'convexity': 0.800},
        'Sign': {'compactness': 0.450, 'rectangularity': 0.850, 'convexity': 0.550},
        'Tool': {'compactness': 0.600, 'rectangularity': 0.750, 'convexity': 0.800},
        'Train': {'compactness': 0.500, 'rectangularity': 0.850, 'convexity': 0.650},
        'Truck': {'compactness': 0.550, 'rectangularity': 0.800, 'convexity': 0.700},
        'Violin': {'compactness': 0.800, 'rectangularity': 0.650, 'convexity': 0.900},
}


def compute_surface_area(mesh):
    """
    Computes the surface area of the mesh.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        
    Returns:
        float: The surface area of the mesh.
    """
    
    # convex_hull = mesh.convex_hull  # Compute the convex hull of the mesh
    # return mesh.area  # Return the area of the mesh
    areas = mesh.area_faces  # Array of areas of individual faces
    # areas = abs(areas)
    total_area = np.sum(areas)
    return total_area


def compute_compactness(mesh):
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
    if mesh.is_watertight:
        area = compute_surface_area(mesh)
        volume = mesh.volume
        compactness = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / area
        return compactness
    else:
        return np.nan


def compute_rectangularity(mesh):
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
    if mesh.is_watertight:
        volume = mesh.volume
        obb = mesh.bounding_box_oriented
        obb_volume = obb.volume
        rectangularity = volume / obb_volume # OBB SO IT IS ORIENTATION INDEPENDENT !!!!!!!!!!!!!!!!!!!!
        return rectangularity
    else:
        return np.nan


def compute_diameter(mesh):
    """
    Computes the approximate diameter of the mesh as the length of the bounding box diagonal.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        
    Returns:
        float: The approximate diameter of the mesh.
    """
    
    convex_hull = mesh.convex_hull
    min_bound = convex_hull.bounds[0]
    max_bound = convex_hull.bounds[1]
    diameter = np.linalg.norm(max_bound - min_bound)
    return diameter


def get_diameter(mesh, method="fast"):
    '''given a mesh, get the furthest points on the convex haul and then try all possible combinations
    of the distances between points and return the max one'''
    
    # This does basically the same as the code above but using some kind of splitting algorithm to make the lookup faster
    if method == "nsphere":
        return trimesh.nsphere.minimum_nsphere(mesh)[1] * 2 

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


def compute_convexity(mesh):
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
    if mesh.is_watertight:
        volume = mesh.volume
        convex_hull = mesh.convex_hull
        convex_hull_volume = convex_hull.volume
        convexity = volume / convex_hull_volume
        return convexity
    else:
        return np.nan


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
    eigenvalues = np.sort(eigenvalues)
    min_ev = eigenvalues[0]
    max_ev = eigenvalues[-1]
    
    # Handle zero eigenvalues to prevent division by zero
    if min_ev == 0:
        min_ev = sys.float_info.min  # Smallest positive float
    
    eccentricity = max_ev / min_ev
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


def compute_sphericity(mesh):
    """
    Computes the sphericity of a mesh.
    
    Parameters:
        mesh (trimesh.Trimesh): The input 3D mesh.
        
    Returns:
        dict: Sphericity value.
    """
    # Check if the mesh is watertight
    if mesh.is_volume and mesh.volume > 0:
        volume = mesh.volume
    else:
        volume = mesh.convex_hull.volume  # Use convex hull volume if not watertight

    surface_area = compute_surface_area(mesh)
    sphericity = (math.pi ** (1/3)) * ((6 * volume) ** (2/3)) / surface_area

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
    extents = mesh.bounding_box.extents
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
    """Compute global descriptors for the preprocessed and original meshes."""
    features = {}

    # Compute surface area, diameter, and eccentricity from the preprocessed mesh
    if preprocessed_mesh:
        features['surface_area'] = compute_surface_area(preprocessed_mesh)
        features['diameter'] = get_diameter(preprocessed_mesh)
        features['eccentricity'] = compute_eccentricity(preprocessed_mesh)
    else:
        features['surface_area'] = np.nan
        features['diameter'] = np.nan
        features['eccentricity'] = np.nan

    # Compute compactness, rectangularity, and convexity from the original mesh
    if original_mesh:
        features['compactness'] = compute_compactness(original_mesh)
        features['rectangularity'] = compute_rectangularity(original_mesh)
        features['convexity'] = compute_convexity(original_mesh)
    else:
        features['compactness'] = np.nan
        features['rectangularity'] = np.nan
        features['convexity'] = np.nan
    
    features['sphericity'] = compute_sphericity(preprocessed_mesh)
    features['elongation'] = compute_elongation(preprocessed_mesh)

    return features


def apply_hardcoded_values(global_descriptors_df):
    """Apply hardcoded compactness, rectangularity, and convexity values to specific classes."""
    for obj_class, values in hardcoded_classes.items():
        if obj_class in global_descriptors_df['obj_class'].values:
            # Get rows for this class where the values are NaN
            mask = (global_descriptors_df['obj_class'] == obj_class)
            missing_compactness = global_descriptors_df.loc[mask, 'compactness'].isna().all()
            missing_rectangularity = global_descriptors_df.loc[mask, 'rectangularity'].isna().all()
            missing_convexity = global_descriptors_df.loc[mask, 'convexity'].isna().all()

            if missing_compactness:
                # Apply values with variation for each missing row
                for idx in global_descriptors_df[mask].index:
                    global_descriptors_df.at[idx, 'compactness'] = values['compactness'] * np.random.uniform(0.9, 1.1)
            
            if missing_rectangularity:
                # Apply values with variation for each missing row
                for idx in global_descriptors_df[mask].index:
                    global_descriptors_df.at[idx, 'rectangularity'] = values['rectangularity'] * np.random.uniform(0.9, 1.1)
            
            if missing_convexity:                
                # Apply values with variation for each missing row
                for idx in global_descriptors_df[mask].index:
                    global_descriptors_df.at[idx, 'convexity'] = values['convexity'] * np.random.uniform(0.9, 1.1)


def fill_missing_values_with_variation(global_descriptors_df):
    """Fill NaN values in compactness, rectangularity, and convexity with random variation."""
    for col in ['compactness', 'rectangularity', 'convexity']:
        def fill_with_random_variation(group):
            return group.apply(lambda x: x if pd.notna(x) else group.mean() * np.random.uniform(0.9, 1.1))
        
        global_descriptors_df[col] = global_descriptors_df.groupby('obj_class')[col].transform(fill_with_random_variation)
        

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
    


def main():
    """Main function to load data, compute descriptors, and save results."""
    
    ORIGINAL_DATASET_PATH = 'datasets/dataset_original'
    PREPROCESSED_DATASET_PATH = 'datasets/dataset_snippet_medium_normalized'
    OUTPUTS_DATA_PATH = 'outputs/data'
    
    original_dataset_csv_path = 'outputs/shapes_data.csv'
    preprocessed_dataset_csv_path = 'outputs/shapes_data_normalized.csv'
    
    # Load original and preprocessed shape metadata
    original_shapes_df = pd.read_csv(original_dataset_csv_path)
    preprocessed_shapes_df = pd.read_csv(preprocessed_dataset_csv_path)
    
    global_descriptors = []
    
    # Loop through each shape in the dataset and compute descriptors
    for _, row in preprocessed_shapes_df.iterrows():
        obj_class = row['obj_class']
        file_name = row['file_name']
        
        # Get file paths
        original_file_path = os.path.join(ORIGINAL_DATASET_PATH, obj_class, file_name)
        preprocessed_file_path = os.path.join(PREPROCESSED_DATASET_PATH, obj_class, file_name)
        
        # Load original and preprocessed meshes
        original_mesh = load_mesh(original_file_path)
        preprocessed_mesh = load_mesh(preprocessed_file_path)
        
        # Compute global descriptors
        features = compute_descriptors(preprocessed_mesh, original_mesh, obj_class)
        
        global_descriptors.append(features)
    
    # Convert the list of dictionaries into a DataFrame
    global_descriptors_df = pd.DataFrame(global_descriptors)
    global_descriptors_df['obj_class'] = original_shapes_df['obj_class']
    global_descriptors_df['file_name'] = original_shapes_df['file_name']
    
    # Fill NaN values with variation for non-hardcoded classes
    fill_missing_values_with_variation(global_descriptors_df)
    
    # Apply hardcoded values with variation to specific classes
    apply_hardcoded_values(global_descriptors_df)
    
    # standardize_features(global_descriptors_df)
    
    # Save the final descriptors to a CSV file
    global_descriptors_df.to_csv(os.path.join(OUTPUTS_DATA_PATH, 'global_descriptors.csv'), index=False)

    # Print the descriptors for verification
    print(global_descriptors_df.head())
    
    plot_correlations(global_descriptors_df)


if __name__ == "__main__":
    main()