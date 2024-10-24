import os
import pandas as pd
import numpy as np
import time

import warnings
warnings.filterwarnings('ignore')

import logging
import trimesh
import multiprocessing as mp
from prep_resampling import create_directory, analyze_dataset, plot_distribution


# Override
OVERWRITE = True

# Number of parallel processors
PROCESSORS = 10

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

# NORMALIZED_SHAPES_PATH = 'datasets/dataset_snippet_small_normalized'
NORMALIZED_SHAPES_PATH = 'datasets/dataset_snippet_medium_normalized'

OUTPUT_PATH = 'outputs'
OUTPUTS_DATA_PATH = 'outputs/data'
OUTPUTS_PLOTS_PATH = 'outputs/plots'


def center_at_origin(mesh):
    """Translates the mesh so that its centroid coincides with the origin."""
    
    translated_mesh = mesh.copy()
    translated_mesh.vertices = mesh.vertices - mesh.centroid
    return translated_mesh


def scale_to_unit(mesh):
    """Scales the mesh so that it fits tightly in a unit-sized cube."""
    scaled_mesh = mesh.copy()
    maxsize = np.max(mesh.bounding_box.extents)  # find max coordinate magnitude in any dimension
    scaled_mesh.apply_scale(1 / maxsize)
    return scaled_mesh


def pca_eigenvectors(mesh, verbose = False):
    """"Return PCA eigenvectors (major variance first, least variance last)"""
    
    # this is a matrix of points of shape (3, nr points)
    A = np.transpose(mesh.vertices)
    A_cov = np.cov(A)
    eigenvalues, eigenvectors = np.linalg.eig(A_cov)
    
    # we now sort eigenvalues by ascending order, saving the index of each rank position:
    ascend_order = np.argsort(eigenvalues)
    
    if verbose: logger.info("PCA before alignment\n", eigenvalues, eigenvectors)
    
    # e1, e2, e3 based on the order of the eigenvalues magnitudes
    # e1, e2, e3 all have magnitude 1
    e3, e2, e1 = (eigenvectors[:,index] for index in ascend_order) # the eigenvectors are the COLUMNS of the vector matrix

    return e1, e2, e3 # we return them in descending order


def pca_align(mesh, verbose=False):
    """Aligns the mesh using PCA so that the largest variance is along the x-axis and the second largest along the y-axis."""
    
    # Calculate PCA eigenvectors
    e1, e2, e3 = pca_eigenvectors(mesh, verbose=verbose)

    # Create a copy of the mesh to store the aligned vertices
    aligned_mesh = mesh.copy()
    aligned_mesh.vertices = np.zeros(mesh.vertices.shape)

    # For each vertex, calculate the new position by projecting onto the PCA eigenvectors
    for index in range(mesh.vertices.shape[0]):
        point = mesh.vertices[index]
        
        # New coordinates are the projections of the original point onto the eigenvectors
        aligned_mesh.vertices[index] = np.dot(point, e1), np.dot(point, e2), np.dot(point, np.cross(e1, e2))
    
    return aligned_mesh


def moments_of_inertia(mesh):
    """Find moments of inertia along the x y and z axes"""

    # Get the centers of the triangles that make up the mesh
    triangles = mesh.triangles_center
    
    # Calculate the moments of inertia (fx, fy, fz) for each axis
    fx, fy, fz = np.sum(
        [ (np.sign(x)*x*x, np.sign(y)*y*y, np.sign(z)*z*z) for x,y,z in triangles],
                        axis = 0)
    
    return fx, fy, fz


def moment_flip(mesh):
    """Flips the mesh based on moments of inertia so the heaviest part is on the positive side of each axis."""
    
    # Calculate moments of inertia along x, y, and z axes
    fx, fy, fz = moments_of_inertia(mesh)

    # Determine the sign of the moments of inertia for each axis
    sx, sy, sz = np.sign([fx, fy, fz])

    # For each vertex, multiply its coordinates by the corresponding sign to flip it
    for index in range(mesh.vertices.shape[0]):
        mesh.vertices[index] = np.multiply(mesh.vertices[index], (sx, sy, sz))

    return mesh


def normalize_input_shape(mesh):
    """
    Process and normalize the input shape in the search engine (mesh).
    
    Parameters:
        row (pd.Series): A row from the DataFrame containing file paths and metadata.
        normalized_root (str): Root directory where the normalized meshes will be saved.
    
    Returns:
        str: The file path if successful, or None if an error occurs.
    """

    try:

        # Apply the normalization steps
        mesh = center_at_origin(mesh)   # Step 1: Center the mesh at the origin
        mesh = scale_to_unit(mesh)      # Step 2: Scale the mesh to fit within a unit cube
        mesh = pca_align(mesh)          # Step 3: Align the mesh using PCA
        mesh = moment_flip(mesh)        # Step 4: Flip the mesh based on moments of inertia
        
        # Ensure normals are consistent and fix them
        if not mesh.is_winding_consistent:
            mesh.fix_normals()

        # Remove degenerate faces (optional, but useful)
        mesh.remove_degenerate_faces()

        return mesh

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None


def normalize_single_shape(row, normalized_root):
    """
    Process and normalize a single shape (mesh).
    
    Parameters:
        row (pd.Series): A row from the DataFrame containing file paths and metadata.
        normalized_root (str): Root directory where the normalized meshes will be saved.
    
    Returns:
        str: The file path if successful, or None if an error occurs.
    """
    obj_class = row['obj_class']
    file_name = row['file_name']
    file_path = row['file_path']
    normalized_file_path = os.path.join(normalized_root, obj_class, file_name)

    try:
        # Load the mesh using Trimesh
        mesh = trimesh.load(file_path)

        # Apply the normalization steps
        mesh = center_at_origin(mesh)   # Step 1: Center the mesh at the origin
        mesh = scale_to_unit(mesh)      # Step 2: Scale the mesh to fit within a unit cube
        mesh = pca_align(mesh)          # Step 3: Align the mesh using PCA
        mesh = moment_flip(mesh)        # Step 4: Flip the mesh based on moments of inertia
        
        # Ensure normals are consistent and fix them
        if not mesh.is_winding_consistent:
            mesh.fix_normals()

        # Remove degenerate faces (optional, but useful)
        mesh.remove_degenerate_faces()

        # Ensure the directory for normalized file exists
        os.makedirs(os.path.dirname(normalized_file_path), exist_ok=True)

        # Save the normalized mesh
        mesh.export(normalized_file_path)

        return normalized_file_path

    except Exception as e:
        logger.error(f"Error processing mesh {file_path}: {str(e)}")
        return None


def normalize_shapes_parallel(df, normalized_root, num_processes=8):
    """
    Parallelize the normalization of shapes across multiple processors.
    
    Parameters:
        df (pd.DataFrame): The dataset containing file paths and metadata.
        normalized_root (str): The root directory where normalized files will be saved.
        num_processes (int): The number of parallel processes to use.
    
    Returns:
        list: List of paths to successfully normalized meshes.
    """
    with mp.Pool(processes=num_processes) as pool:
        # Parallelize the normalization of each mesh row
        results = pool.starmap(normalize_single_shape, [(row, normalized_root) for _, row in df.iterrows()])

    # Filter out any None values (i.e., failed meshes)
    successful_paths = [res for res in results if res is not None]
    
    return successful_paths


if __name__ == "__main__":
    
    # Create directories and CSV paths 
    create_directory(NORMALIZED_SHAPES_PATH, overwrite=OVERWRITE, logger=logger)
    normalized_csv_path = os.path.join(OUTPUT_PATH, "shapes_data_normalized.csv")
    times_path = os.path.join(OUTPUTS_DATA_PATH, "times.csv")
    
    times_df = pd.read_csv(os.path.join(OUTPUTS_DATA_PATH, "times.csv"))
    
    normalization_time = {}
    start_normalization = time.time()
    
    # --------------------------------------------------- NORMALIZE SHAPES ---------------------------------------------------- #    
    if not os.path.exists(normalized_csv_path) or OVERWRITE:
        logger.info("# Step 14: Normalize shapes...")
        
        remeshed_shapes_df = pd.read_csv('outputs/shapes_data_remeshed.csv')
        normalize_shapes_parallel(remeshed_shapes_df, NORMALIZED_SHAPES_PATH, num_processes=PROCESSORS)
        
        normalized_data = analyze_dataset(NORMALIZED_SHAPES_PATH)
        normalized_shapes_df = pd.DataFrame(normalized_data)
        
        normalized_shapes_df.to_csv(normalized_csv_path, index=False)
        # logger.info(f"-----> Normalized shapes data saved to '{normalized_csv_path}'")

        plot_distribution(normalized_shapes_df, "Normalized distribution", OUTPUTS_PLOTS_PATH)
        
    end_normalization = time.time()
    
    times_df['normalization'] = end_normalization - start_normalization
    
    times_df.to_csv(times_path, index=False)
    