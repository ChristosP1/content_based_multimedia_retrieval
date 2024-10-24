import os
import pymeshlab.pmeshlab
import trimesh
import pymeshlab
import tempfile
import uuid 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pyemd import emd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


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
def z_score_normalize_input(global_descriptors_df, normalization_params_path):
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


##################################################################################################################
##################################################################################################################
def compute_cosine_similarity(global_desc_df, input_mesh_global_descriptor_df):
    """
    Compute cosine similarity between the global descriptors of a given mesh
    and all the meshes in the dataset.
    
    Parameters:
        global_desc_df (pd.DataFrame): DataFrame containing global descriptors of all meshes.
        mesh_global_descriptor_df (pd.DataFrame): DataFrame containing global descriptors of the input mesh.
        
    Returns:
        pd.Series: Cosine similarity scores between the given mesh and all dataset meshes.
    """
    
    # Preserve the object class and file name for later reference
    obj_metadata = global_desc_df[['file_name', 'obj_class']]
    
    # Drop the columns irrelevant for similarity computation
    global_desc_df = global_desc_df.drop(columns=['obj_class', 'file_name'])
    
    # Convert to numpy array
    global_desc_data = global_desc_df.to_numpy()
    
    # Extract the single mesh global descriptors as a numpy array
    mesh_descriptor = input_mesh_global_descriptor_df.to_numpy().reshape(1, -1)

    # Sanity check on dimensions (optional, can be removed later)
    print(f"Input descriptor shape: {input_mesh_global_descriptor_df.shape}")
    print(f"Dataset descriptor shape: {global_desc_df.shape}")
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(mesh_descriptor, global_desc_data)
    
    # Return the similarity scores along with the file names as a DataFrame
    similarity_df = pd.DataFrame({
        'file_name': obj_metadata['file_name'], 
        'class': obj_metadata['obj_class'],
        'similarity': cosine_sim[0]
    })

    # Sort the DataFrame by similarity scores in descending order
    similarity_df = similarity_df.sort_values(by='similarity', ascending=False).reset_index(drop=True).round(3)


    # Return cosine similarity as a Pandas Series
    return similarity_df


def compute_emd_similarity(input_local_desc, dataset_local_desc):
    """
    Compute EMD (Earth Mover's Distance) similarity between input and dataset objects based on local descriptors.
    
    Parameters:
        input_local_desc (pd.DataFrame): DataFrame containing local descriptors (A3, D1, D2, D3, D4) for the input object.
        dataset_local_desc (pd.DataFrame): DataFrame containing local descriptors (A3, D1, D2, D3, D4) for dataset objects.
        
    Returns:
        pd.DataFrame: DataFrame with EMD scores for each object in the dataset.
    """
    # Preserve the object class and file name for reference
    obj_metadata = dataset_local_desc[['file_name', 'obj_class']]
    
    print(type(input_local_desc['A3'].values[0]))
    # Extract the histograms of the input object (as lists)
    input_histograms = {
        'A3': input_local_desc['A3'].values[0],
        'D1': input_local_desc['D1'].values[0],
        'D2': input_local_desc['D2'].values[0],
        'D3': input_local_desc['D3'].values[0],
        'D4': input_local_desc['D4'].values[0]
    }
    
    emd_scores = []
    
    # Loop through each object in the dataset and compute EMD for each local descriptor
    for index, row in dataset_local_desc.iterrows():
        dataset_histograms = {
            'A3': eval(row['A3']),
            'D1': eval(row['D1']),
            'D2': eval(row['D2']),
            'D3': eval(row['D3']),
            'D4': eval(row['D4'])
        }
        
        # Compute EMD for each histogram
        emd_A3 = wasserstein_distance(input_histograms['A3'], dataset_histograms['A3'])
        emd_D1 = wasserstein_distance(input_histograms['D1'], dataset_histograms['D1'])
        emd_D2 = wasserstein_distance(input_histograms['D2'], dataset_histograms['D2'])
        emd_D3 = wasserstein_distance(input_histograms['D3'], dataset_histograms['D3'])
        emd_D4 = wasserstein_distance(input_histograms['D4'], dataset_histograms['D4'])
        
        total_emd = (emd_A3 + emd_D1 + emd_D2 + emd_D3 + emd_D4) / 5
        # Combine the EMD results (here we use a simple average, but you can use a weighted sum)
        total_emd = 0.2 * emd_A3 + 0.2 * emd_D1 + 0.2 * emd_D2 + 0.2 * emd_D3 + 0.2 * emd_D4  # Example of weighted sum


        
        # Append the result to the EMD scores list
        emd_scores.append(total_emd)
    
    # Return a DataFrame with EMD scores along with file names and classes
    similarity_df = pd.DataFrame({
        'file_name': obj_metadata['file_name'],
        'class': obj_metadata['obj_class'],
        'emd_score': emd_scores
    })
    
    # Sort the DataFrame by EMD score (lower EMD = more similar)
    similarity_df = similarity_df.sort_values(by='emd_score', ascending=True).reset_index(drop=True).round(3)
    
    return similarity_df


def compute_combined_similarity(global_similarity_df, local_similarity_df, global_weight=0.4, local_weight=0.6):
    """
    Combine global and local similarities with specified weights.
    
    Parameters:
        global_similarity_df (pd.DataFrame): DataFrame containing global similarity scores.
        local_similarity_df (pd.DataFrame): DataFrame containing local similarity scores (EMD scores).
        global_weight (float): Weight for the global similarity (default: 0.4).
        local_weight (float): Weight for the local similarity (default: 0.6).
        
    Returns:
        pd.DataFrame: Final DataFrame containing combined similarity scores.
    """
    # Merge global and local similarity DataFrames on 'file_name'
    combined_df = pd.merge(global_similarity_df, local_similarity_df, on='file_name', how='inner', suffixes=('_global', '_local'))
    
    # Normalize EMD scores to turn them into similarity scores (lower EMD = higher similarity)
    combined_df['emd_similarity'] = 1 / (1 + combined_df['emd_score'])
    
    # Compute weighted similarity
    combined_df['final_similarity'] = global_weight * combined_df['similarity'] + local_weight * combined_df['emd_similarity']
    
    # Sort by the final similarity score in descending order
    combined_df = combined_df.sort_values(by='final_similarity', ascending=False).reset_index(drop=True)
    
    return combined_df[['file_name', 'class_global', 'similarity', 'emd_similarity', 'final_similarity']]


##################################################################################################################
##################################################################################################################


def preprocess_input_mesh(mesh_path):
    try:
        # 1. Read mesh as trimesh file
        mesh = trimesh.load(mesh_path)
        print(f"- Initial shape: Vertices: {len(mesh.vertices)}  |  Faces: {len(mesh.faces)}")
        
        # 2. Clean, Resample and again clean the mesh
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
        input_mesh_global_descriptors_df = z_score_normalize_input(input_mesh_global_descriptors_df, 'outputs/data/standardization_params_z_score.csv')
        print(input_mesh_global_descriptors_df.head())
        
        # 6. Compute Local descriptors
        avg_bins = pd.read_csv('outputs/data/average_bins_local.csv')
        print(avg_bins)
        input_mesh_local_descriptors_df = pd.DataFrame([compute_local_descriptors(mesh, 4000, avg_bins)])
        input_mesh_local_descriptors_df = input_mesh_local_descriptors_df.round(4)
        input_mesh_local_descriptors_df.to_csv("aaa.csv")

        return input_mesh_global_descriptors_df, input_mesh_local_descriptors_df
        
        
    except ValueError as e:
        print(f"Error: {e}")
        

def find_similar_objects(input_global_desc_df, input_local_desc_df):
    
    
    # 7. Similarity of Global descriptors
    print("Find global similarity")
    global_descriptors_df = pd.read_csv('outputs/data/global_descriptors_standardized.csv')
    global_similarity_df = compute_cosine_similarity(global_descriptors_df, input_global_desc_df)
    
    # 8. Similarity of Local descriptors
    print("Find local similarity")
    local_descriptors_df  = pd.read_csv('outputs/data/local_descriptors.csv')
    local_similarity_df = compute_emd_similarity(input_local_desc_df, local_descriptors_df)
    
        
    # Get the top 5 most similar meshes
    top_similar_meshes_global = global_similarity_df.head(10)
    top_similar_meshes_local = local_similarity_df.head(10)
    
    final_similarity_df = compute_combined_similarity(global_similarity_df, local_similarity_df, global_weight=0.5, local_weight=0.5)
    top_similar_meshes = final_similarity_df.head(10)
    
    
    return top_similar_meshes_global, top_similar_meshes_local, top_similar_meshes


if __name__ == "__main__":
    preprocess_input_mesh("datasets/dataset_medium/Biplane/D00132.obj")



