import os
import pandas as pd
import matplotlib.pyplot as plt
import pymeshlab
import math
import vedo
import logging
from tqdm import tqdm
import shutil
import multiprocessing
from multiprocessing import Pool, cpu_count
import time
import queue
from datetime import datetime



# Override
OVERWRITE = True

# Outlier thresholds
LOW_THRESHOLD = 500
HIGH_THRESHOLD = 10000

# Tolerance level
TOLERANCE = 0.2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

# Dataset and Output paths
DATASET_PATH = 'datasets/dataset_snippet_small'
RESAMPLED_SHAPES_PATH = 'datasets/dataset_snippet_small_resample'
OUTPUT_PATH = 'outputs'

# Function to handle directory creation with optional overwrite
def create_directory(path, overwrite=False, logger=logger):
    """
    Create a directory. If overwrite is True, delete the directory if it exists and recreate it.
    Parameters:
        path (str): The path of the directory to create.
        overwrite (bool): Whether to overwrite the directory if it exists.
    """
    if overwrite and os.path.exists(path):
        # Remove the directory and its contents if overwrite is True
        shutil.rmtree(path)
        logger.info(f"Directory '{path}' existed and was removed for overwriting.")
    
    # Now create the directory
    os.makedirs(path, exist_ok=True)
    logger.info(f"Directory '{path}' created.")




# Counter of shapes excluded
EXCLUDED = 0



def get_face_type(mesh):
    """
    Determine the type of faces in the mesh.

    Parameters:
        mesh (vedo.Mesh): The mesh object to analyze.

    Returns:
        str: A string describing the face type ("Triangles", "Quads", or "Mixed").
    """
    # Get the faces from the mesh
    face_array = mesh.cells  # Renaming the variable to avoid conflict
    face_sizes = [len(face) for face in face_array]

    if all(size == 3 for size in face_sizes):
        return "Triangles"
    elif all(size == 4 for size in face_sizes):
        return "Quads"
    else:
        return "Mixed"

    

def analyze_shape(filepath, low_threshold=500, high_threshold=10000):
    """
    Analyze a 3D shape using Vedo and return the required information.

    Parameters:
        filepath (str): Path to the OBJ file.
        low_threshold (int, optional): Threshold for low vertex count outliers. Defaults to 500.
        high_threshold (int, optional): Threshold for high vertex count outliers. Defaults to 10000.
    Returns:
        dict: A dictionary containing shape analysis results.
    """
    # logger.info(f"Analyzing shape: '{filepath}'")
    try:
        # Load the mesh from the file
        mesh = vedo.Mesh(filepath)

        # Ensure the loaded object is a Mesh
        if not isinstance(mesh, vedo.Mesh):
            logger.warning(f"File '{filepath}' is not a valid mesh. Skipping.")
            return None

        # Get number of vertices and faces
        num_vertices = mesh.npoints
        num_faces = mesh.ncells
        
        # Get face type
        face_type = get_face_type(mesh)

        # Get bounding box
        bounding_box = mesh.bounds()

        # Determine if mesh has holes (Vedo doesn't provide a direct method, so we keep it hardcoded)
        has_holes = False  # Placeholder, you might need to implement a method to check for holes

        # Determine outlier status based on vertex count
        outlier_low = num_vertices <= low_threshold
        outlier_high = num_vertices >= high_threshold

        # Log detailed information
        logger.debug(f"Vertices: {num_vertices}, Faces: {num_faces}, Face Type: {face_type}")

        return {
            "vertices": num_vertices,
            "faces": num_faces,
            "face_type": face_type,
            "bounding_box": {
                "min_bound": [bounding_box[0], bounding_box[2], bounding_box[4]],
                "max_bound": [bounding_box[1], bounding_box[3], bounding_box[5]]
            },
            "has_holes": has_holes,
            "outlier_low": outlier_low,
            "outlier_high": outlier_high
        }

    except Exception as e:
        logger.error(f"Error analyzing shape '{filepath}': {e}")
        return None


def analyze_database(db_path):
    """
    Iteratively analyze all 3D shapes within the database.

    Parameters:
        labeled_db_path (str): Path to the root of the labeled database.
    Returns:
        list: A list of dictionaries containing shape analysis results.
    """
    logger.info(f"Starting analysis of database at '{db_path}'")
    results = []

    for category in os.listdir(db_path):
        category_path = os.path.join(db_path, category)

        if os.path.isdir(category_path):
            # logger.info(f"Processing category: '{category}'")
            for shape_file in os.listdir(category_path):
                if shape_file.lower().endswith('.obj'):
                    filepath = os.path.join(category_path, shape_file)
                    shape_info = {
                        "file_name": shape_file,
                        "obj_class": category,
                        "file_path": filepath
                    }
                    analysis_result = analyze_shape(filepath, high_threshold=HIGH_THRESHOLD, low_threshold=LOW_THRESHOLD)
                    if analysis_result:
                        shape_info.update(analysis_result)
                        results.append(shape_info)
                    else:
                        logger.warning(f"Skipping file '{filepath}' due to previous errors.")

    logger.info("Database analysis complete.")
    return results



# Visualize
def plot_distribution(df, title, bins=30):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(df['vertices'], bins=bins, color='purple', alpha=0.7)
    plt.title(f'{title} - Distribution of Vertices')
    plt.xlabel('Number of Vertices')
    plt.ylabel('Number of Shapes')
    plt.subplot(1, 2, 2)
    plt.hist(df['faces'], bins=bins, color='red', alpha=0.7)
    plt.title(f'{title} - Distribution of Faces')
    plt.xlabel('Number of Faces')
    plt.ylabel('Number of Shapes')
    plt.tight_layout()
    plt.show()

# Statistics
def compare_statistics(original_df, resampled_df):
    print("Statistics comparison:")
    for column in ['vertices', 'faces']:
        print(f"\n{column.capitalize()}:")
        print("Before Resampling:")
        print(original_df[column].describe())
        print("\nAfter Resampling:")
        print(resampled_df[column].describe())




def preprocess_mesh(tasks_to_accomplish, tasks_that_are_done):
    k = 14331**333
    print(k)




def main():
    startTime = datetime.now()

    # Create OUTPUT, RESAMPLED_SHAPES directories and CSV paths 
    create_directory(OUTPUT_PATH, overwrite=OVERWRITE, logger=logger)
    create_directory(RESAMPLED_SHAPES_PATH, overwrite=OVERWRITE, logger=logger)

    original_csv_path = os.path.join(OUTPUT_PATH, "shapes_data.csv")
    resampled_csv_path = os.path.join(OUTPUT_PATH, "resampled_shapes_data.csv")

    # -------------------------------------------- ANALYZE ORIGINAL SHAPES -------------------------------------------- #
    if not os.path.exists(original_csv_path) or OVERWRITE:
        logger.info("# Step 1. # Analyzing database and collecting statistics...")
        shapes_data = analyze_database(DATASET_PATH)
        shapes_data_df = pd.DataFrame(shapes_data)
        shapes_data_df.to_csv(original_csv_path, index=False)
        logger.info(f"Original shapes data saved to '{original_csv_path}'")
    else:
        shapes_data_df = pd.read_csv(original_csv_path)
        logger.info(f"# Step 1. # Loaded existing original shapes data from '{original_csv_path}'")

    # plot_distribution(shapes_data_df, "Before Resampling")
    
    # ----------------------------------------------- RESAMPLING SHAPES ----------------------------------------------- #
    if not os.path.exists(RESAMPLED_SHAPES_PATH) or OVERWRITE:
        logger.info("# Step 2. # Resampling of shapes...")
        shapes_data_df = pd.read_csv('outputs/shapes_data.csv')
        mesh_files = shapes_data_df['file_path'].tolist()
        print(f"Number of mesh files {len(mesh_files)}")
        tasks = {file:i for i, file in enumerate(mesh_files)}
        
        # Use Manager for shared Queues
        number_of_processes = 1
        print(f"Number of processes: {number_of_processes}")
        
        pool = Pool(number_of_processes)
        
        
        
        for key, value in tasks.items():
            pool.apply_async(preprocess_mesh, args=(key, value))
        
        pool.close()
        pool.join()
        

        print(datetime.now() - startTime)

if __name__ == "__main__":
    main()

































