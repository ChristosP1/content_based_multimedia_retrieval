import os
import pymeshlab.pmeshlab
import trimesh
import pymeshlab
import tempfile
import uuid 
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean


from prep_resampling import clean_mesh, adjust_mesh_complexity
from prep_remeshing import isotropic_remesh
from prep_normalizing import *
from desc_global import *
from desc_local import *



##################################################################################################################
##################################################################################################################
def trimesh_to_pymeshlab(trimesh_mesh):
    # Create a temporary file to save the Trimesh object as an OBJ file
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as temp_obj_file:
        # Save Trimesh to the temp OBJ file
        trimesh_mesh.export(temp_obj_file.name, file_type='obj')
        # Load the OBJ file into PyMeshLab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_obj_file.name)
        
    # Remove the temporary file (since PyMeshLab has loaded the mesh)
    os.remove(temp_obj_file.name)
    
    return ms

def pymeshlab_to_trimesh(ms, mesh_index=0):
    # Create a temporary file to save the PyMeshLab mesh as an OBJ file
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as temp_obj_file:
        # Save the PyMeshLab mesh to the temp OBJ file  
        ms.save_current_mesh(temp_obj_file.name, save_face_color=False)
        # Load the OBJ file into Trimesh
        remeshed_trimesh = trimesh.load(temp_obj_file.name)
    
    # Remove the temporary file (since Trimesh has loaded the mesh)
    os.remove(temp_obj_file.name)
    
    return remeshed_trimesh


##################################################################################################################
##################################################################################################################
def z_score_normalize_desc(global_descriptors_df, normalization_params_path):
    """
    Normalize the input object's global descriptors using saved normalization parameters.
    
    Parameters:
        global_descriptors_df (pd.DataFrame): DataFrame containing the global descriptors of the input object.
        normalization_params_path (str): Path to the CSV file containing saved normalization parameters.
        
    Returns:
        pd.DataFrame: The standardized global descriptors DataFrame.
    """
    # Load normalization parameters
    normalization_params_df = pd.read_csv(normalization_params_path)
    
    # Apply z-score normalization to the input object's global descriptors
    for _, row in normalization_params_df.iterrows():
        feature = row["feature"]
        mean = row["mean"]
        std = row["std"]
        
        # Normalize the input object's feature using the loaded mean and std
        if std > 0:
            global_descriptors_df[feature] = (global_descriptors_df[feature] - mean) / std
    
    return global_descriptors_df


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
        bins = local_df[descriptor]
        bins_df = pd.DataFrame(bins.tolist(), columns=[f'{descriptor}_{i+1}' for i in range(len(bins[0]))])
        expanded_columns.append(bins_df)

    # Concatenate the expanded columns into a single dataframe
    expanded_df = pd.concat(expanded_columns, axis=1)

    # Return expanded local descriptor dataframe
    return expanded_df


def merge_global_and_local_descriptors(global_descriptors_df, local_descriptors_df, output_file):
    """
    Merges global and expanded local descriptors into a single dataframe and saves the result as a CSV file.
    
    Parameters:
        global_desc_file (str): Path to the global descriptors CSV file.
        local_desc_file (str): Path to the local descriptors CSV file.
        output_file (str): Path to the output CSV file for saving the merged descriptors.
    """

    # Debug: Print shapes of both dataframes before merging
    print(f"- Global Descriptors Shape: {global_descriptors_df.shape}")
    print(f"- Local Descriptors Shape: {local_descriptors_df.shape}")
    
    # Expand the local descriptor histograms into individual columns
    expanded_local_descriptors_df = expand_local_descriptors(local_descriptors_df)

    # Concatenate file_name + obj_class, global descriptors, and expanded local descriptors
    merged_df = pd.concat([global_descriptors_df, expanded_local_descriptors_df], axis=1).reset_index(drop=True).round(4)

    # Debug: Print shapes after merging
    print(f"- Merged Descriptors Shape: {merged_df.shape}")
    
    # Save the merged dataframe to a CSV file
    merged_df.to_csv(output_file, index=False)
    
    return merged_df
    

##################################################################################################################
##################################################################################################################




##################################################################################################################
##################################################################################################################


def preprocess_input_mesh(mesh_path):
    try:
        # 1. Read mesh as trimesh file
        mesh = trimesh.load(mesh_path)
        print(f"- Initial shape: Vertices: {len(mesh.vertices)}  |  Faces: {len(mesh.faces)}")
        
        # 2. Clean, Resample and clean the mesh
        mesh = clean_mesh(mesh)
        mesh = adjust_mesh_complexity(mesh)
        mesh = clean_mesh(mesh)
        print(f"- After resampling: Vertices: {len(mesh.vertices)}  |  Faces: {len(mesh.faces)}")        
        
        # 3. Remesh the mesh using isotropic remeshing + Cleaning
        ms = trimesh_to_pymeshlab(mesh)
        isotropic_remesh(ms)
        mesh = pymeshlab_to_trimesh(ms)
        mesh = clean_mesh(mesh)
        print(f"- After remeshing: Vertices: {len(mesh.vertices)}  |  Faces: {len(mesh.faces)}")
        
        # 4. Normalize mesh
        mesh = normalize_input_shape(mesh)
        print(f"- After normalization: Vertices: {len(mesh.vertices)}  |  Faces: {len(mesh.faces)}")
        
        # 5. Compute Global descriptors
        input_mesh_global_descriptors_df = pd.DataFrame.from_dict([compute_descriptors(mesh, f"{uuid.uuid1()}")])
        # print(input_mesh_global_descriptors_df.head())
        input_mesh_global_descriptors_df = z_score_normalize_desc(input_mesh_global_descriptors_df, 'outputs/data/standardization_params_z_score.csv')
        input_mesh_global_descriptors_df.round(4)
        # print(input_mesh_global_descriptors_df.head())
        
        # 6. Compute Local descriptors
        avg_bins = pd.read_csv('outputs/data/average_bins_local.csv')
        input_mesh_local_descriptors_df = pd.DataFrame([compute_local_descriptors(mesh, 4000, avg_bins)])
                
        merged_df = merge_global_and_local_descriptors(input_mesh_global_descriptors_df, 
                                           input_mesh_local_descriptors_df, 
                                           "outputs/data/input_mesh_combined_descriptors.csv")
        
        return merged_df
        
    except ValueError as e:
        print(f"Error: {e}")


##################################################################################################################
##################################################################################################################


def standardize_and_save_similarity_scores(distances_df, global_weight=0.3, local_weight=0.7):
    """
    Standardize the descriptor distances using the distance weighting (mean and std) 
    and combine them into a final similarity score.
    """
    # distances_file = "outputs/data/similarity_scores.csv"
    distance_weights_file = "outputs/data/distance_weighting_params.csv"
    final_distances_file = "outputs/data/FINAL_distances.csv"
    
    # distances_df = pd.read_csv(distances_df)
    weighting_params = pd.read_csv(distance_weights_file)
    
    standardized_distances = pd.DataFrame()
    standardized_distances['file_name'] = distances_df['file_name']
    standardized_distances['obj_class'] = distances_df['obj_class']
    
    global_descriptors = ["volume", "surface_area", "diameter", "eccentricity", "compactness", 
                          "rectangularity", "convexity", "sphericity", "elongation"]
    local_descriptors = ["A3", "D1", "D2", "D3", "D4"]
    
    standardized_distances_global = []
    standardized_distances_local = []

    for descriptor in weighting_params['descriptor']:
        if descriptor in distances_df.columns:
            mean = weighting_params.loc[weighting_params['descriptor'] == descriptor, 'mean'].values[0]
            std_dev = weighting_params.loc[weighting_params['descriptor'] == descriptor, 'std'].values[0]
            
            standardized_distances[descriptor] = (distances_df[descriptor] - mean) / std_dev
            
            if descriptor in global_descriptors:
                standardized_distances_global.append(standardized_distances[descriptor] * global_weight)
            elif descriptor in local_descriptors:
                standardized_distances_local.append(standardized_distances[descriptor] * local_weight)
    
    standardized_distances['global_score'] = sum(standardized_distances_global)
    standardized_distances['local_score'] = sum(standardized_distances_local)
    
    standardized_distances['final_score'] = standardized_distances['global_score'] + standardized_distances['local_score']
    
    standardized_distances = standardized_distances.sort_values(by='final_score').reset_index(drop=True)
    
    standardized_distances.to_csv(final_distances_file, index=False)
    print(f"Standardized and weighted similarity scores saved to {final_distances_file}")
    
    return standardized_distances
    

def compute_distances(input_df):
    dataset_file = "outputs/data/combined_descriptors.csv"
    distances_file = "outputs/data/similarity_scores.csv"
    
    # input_df = pd.read_csv(input_file_desc)
    dataset_df = pd.read_csv(dataset_file)
    
    global_features = ["volume", "surface_area", "diameter", "eccentricity", "compactness", 
                       "rectangularity", "convexity", "elongation"]
    histograms = ["A3", "D1", "D2", "D3", "D4"]
    
    # Filter global descriptors based on their presence in the input_df and dataset_df
    available_global_features = [feature for feature in global_features if feature in input_df.columns and feature in dataset_df.columns]
    
    input_global = input_df[available_global_features]
    input_local = input_df.drop(columns=global_features, errors='ignore')
    
    results = pd.DataFrame({
        'file_name': dataset_df['file_name'],
        'obj_class': dataset_df['obj_class']
    })
    
    for feature in available_global_features:
        input_value = input_global[feature].values[0]
        results[feature] = dataset_df.apply(lambda row: euclidean([input_value], [row[feature]]), axis=1)

    bin_counts = pd.read_csv('outputs/data/average_bins_local.csv').iloc[0].to_dict()

    for hist in histograms:
        if all(f"{hist}_{i+1}" in input_local.columns for i in range(int(bin_counts[hist]))):
            num_bins = int(bin_counts[hist])
            input_hist_values = input_local[[f"{hist}_{i+1}" for i in range(num_bins)]].values.flatten()
            
            results[hist] = dataset_df.apply(lambda row: wasserstein_distance(
                input_hist_values,
                row[[f"{hist}_{i+1}" for i in range(num_bins)]].values.flatten()
            ), axis=1)

    results = results.round(4)
    results.to_csv(distances_file, index=False)
    print(f"Distances calculated and saved to {distances_file}")
    
    return results


def get_dominant_class_items(similar_meshes_df, dominant_class_name):
    """
    Filter and return all items of the dominant class from the similar meshes dataframe.
    
    Parameters:
        similar_meshes_df (pd.DataFrame): DataFrame containing similar meshes with
                                          'obj_class' as one of the columns.
        dominant_class_name (str): Name of the dominant class to filter on.
    
    Returns:
        pd.DataFrame: A DataFrame containing only items from the dominant class.
    """
    # Filter the dataframe for rows matching the dominant class
    class_items_df = similar_meshes_df[similar_meshes_df['obj_class'] == dominant_class_name]
    
    return class_items_df



def find_dominant_class(standardized_distances, top_n=10):
    """
    Compute the most dominant class from the top N retrieved objects by summing the final
    scores of each class and selecting the one with the lowest cumulative score.
    
    Parameters:
        standardized_distances (pd.DataFrame): DataFrame with standardized distances, 
                                               including 'file_name', 'obj_class', 
                                               and 'final_score' columns.
        top_n (int): Number of top objects to consider (default is 10).
        
    Returns:
        str: The most dominant class.
    """
    # Select the top N objects based on final score
    top_results = standardized_distances.head(top_n)
    
    # Group by class and sum the final scores within each class
    class_scores = top_results.groupby('obj_class')['final_score'].sum()
    print(class_scores)
    sorted_scores = class_scores.sort_values()
    
    most_dominant_class = sorted_scores.index[0]
    second_dominant_class = sorted_scores.index[1] if len(sorted_scores) > 1 else None
    
    dominant_class_score = sorted_scores.iloc[0]
    second_class_score = sorted_scores.iloc[1] if second_dominant_class else float('inf')
    print(dominant_class_score)
    print(second_class_score)
    
     # Check if the most dominant class has at least twice the "dominance" (lower score) compared to the second
    if dominant_class_score <= 2 * second_class_score:
        print(dominant_class_score, " < ", 2 * second_class_score)
        return most_dominant_class
    else:
        return False



if __name__ == "__main__":
    # File paths
    input_desc = "outputs/data/input_mesh_combined_descriptors.csv"
    dataset_desc = "outputs/data/combined_descriptors.csv"
    distances_file = "outputs/data/similarity_scores.csv"
    distance_weights_file = "outputs/data/distance_weighting_params.csv"
    final_distances_file = "outputs/data/FINAL_distances.csv"
    
    # Preprocess input and compute Global and Local descriptors
    # preprocess_input_mesh("datasets/dataset_snippet_medium/Hand/m327.obj")
    # preprocess_input_mesh("datasets/dataset_snippet_medium/Car/D00377.obj")
    merged_df = preprocess_input_mesh("datasets/dataset_snippet_medium/Bottle/m482.obj")
    
    # Compute distances
    distances = compute_distances(merged_df)
    
    # Standardize dinstances
    standardize_and_save_similarity_scores(distances)
    
    



