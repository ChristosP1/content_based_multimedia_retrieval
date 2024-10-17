import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend for PNGs
import matplotlib.pyplot as plt
import time

import warnings
warnings.filterwarnings('ignore')

import logging
import shutil
import trimesh
from copy import deepcopy
import multiprocessing as mp


# Override
OVERWRITE = True

# Outlier thresholds
LOW_THRESHOLD = 500
HIGH_THRESHOLD = 10000


# Target vertices
TARGET_VERTICES = 5000

# Tolerance level
TOLERANCE = 0.2

# Number of parallel processors
PROCESSORS = 10

# Upper and lower bounds
LOWER_BOUND = TARGET_VERTICES-TARGET_VERTICES*TOLERANCE
UPPER_BOUND = TARGET_VERTICES+TARGET_VERTICES*TOLERANCE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

# Dataset and Output paths
OUTPUT_PATH = 'outputs'
OUTPUTS_DATA_PATH = 'outputs/data'
OUTPUTS_PLOTS_PATH = 'outputs/plots'

# DATASET_PATH = 'datasets/dataset_snippet_small'
# RESAMPLED_SHAPES_PATH = 'datasets/dataset_snippet_small_resampled'

DATASET_PATH = 'datasets/dataset_snippet_medium'
CLEANED_SHAPES_PATH = 'datasets/dataset_snippet_medium_cleaned'
RESAMPLED_SHAPES_PATH = 'datasets/dataset_snippet_medium_resampled'
CLEANED_RESAMPLED_SHAPES_PATH = 'datasets/dataset_snippet_medium_resampled_cleaned'



# Function to handle directory creation with optional overwrite
def create_directory(path, overwrite=False, logger=logger):
    """
    Create a directory. If overwrite is True, delete the directory if it exists and recreate it.
    
    Parameters:
        path (str): The path of the directory to create.
        overwrite (bool): Whether to overwrite the directory if it exists.
    """
    # if overwrite and os.path.exists(path):
        # Remove the directory and its contents if overwrite is True
        # shutil.rmtree(path)
        # logger.info(f"Directory '{path}' existed and was removed for overwriting.")
    
    # Now create the directory
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    # logger.info(f"Directory '{path}' created.")


def analyze_shape(filepath=False, mesh=None):
    '''
    Analyze a 3D shape using Trimesh and return the required information.
    :param filepath: Path to the OBJ file
    '''
    
    try:
        # Load the mesh using Trimesh
        if filepath:
            mesh = trimesh.load(filepath)
        
        num_vertices = len(mesh.vertices)  # Number of vertices
        num_faces = len(mesh.faces)  # Number of faces (triangles)
        face_type = "Triangles"  # Face type (triangles by default for OBJ)
        
        # Axis-aligned bounding box (AABB)
        aabb_min, aabb_max = mesh.bounds
        bounding_box = {
            "min_bound": aabb_min.tolist(),  # Min corner of the bounding box
            "max_bound": aabb_max.tolist()   # Max corner of the bounding box
        }

        is_manifold = mesh.is_watertight  # Has holes or not
        outlier_low = True if num_vertices <= LOW_THRESHOLD else False  # Low outlier
        outlier_high = True if num_vertices >= HIGH_THRESHOLD else False  # High outlier

        # Return dictionary of extracted data
        return {
            "vertices": num_vertices,
            "faces": num_faces,
            "face_type": face_type,
            "bounding_box": bounding_box,
            "is_manifold": is_manifold,
            "outlier_low": outlier_low,
            "outlier_high": outlier_high
        }
        
    except Exception as e:
        print(f"Error analyzing shape {filepath}: {str(e)}")
        return None


def analyze_dataset(labeled_db_path):
    '''
    Iteratively analyze all 3D shapes within the database.
    :param labeled_db_path: Path to the root of the labeled database
    '''
    results = []

    for category in os.listdir(labeled_db_path):
        category_path = os.path.join(labeled_db_path, category)

        if os.path.isdir(category_path):
            for shape_file in os.listdir(category_path):
                if shape_file.endswith('.obj'):
                    filepath = os.path.join(category_path, shape_file)
                    shape_info = {}
                    shape_info["file_name"] = shape_file  # Add the filename of the shape
                    shape_info["obj_class"] = category  # Add class (category) of the shape
                    # Analyze each shape
                    shape_info.update(analyze_shape(filepath))
                    shape_info["file_path"] = os.path.join(category_path, shape_file)  # Add class (category) of the shape
                    
                    results.append(shape_info)

    return results


def compute_and_save_statistics(shapes_data_df, statistics_csv_path='outputs/data/global_statistics.csv', classes_txt_path='outputs/data/classes.txt'):
    """
    Compute statistics on the vertices and faces columns, and save the global statistics and unique object classes to files.

    Parameters:
        shapes_data_df (pd.DataFrame): The DataFrame containing the mesh data.
        statistics_csv_path (str): The path to save the global statistics CSV file.
        classes_txt_path (str): The path to save the list of unique classes.
    """
    # Compute mean and standard deviation on both vertices and faces
    mean_vertices = shapes_data_df['vertices'].mean()
    std_vertices = shapes_data_df['vertices'].std()

    mean_faces = shapes_data_df['faces'].mean()
    std_faces = shapes_data_df['faces'].std()

    # Compute other statistics
    total_shapes = shapes_data_df.shape[0]
    is_manifold_count = shapes_data_df[shapes_data_df['is_manifold'] == True]['is_manifold'].count() 
    outlier_low_count = shapes_data_df[shapes_data_df['outlier_low'] == True]['outlier_low'].count()
    outlier_high_count = shapes_data_df[shapes_data_df['outlier_high'] == True]['outlier_high'].count()

    # Prepare global statistics
    global_statistics = {
        "total_shapes": total_shapes,
        "mean_vertices": mean_vertices,
        "std_vertices": std_vertices,
        "mean_faces": mean_faces,
        "std_faces": std_faces,
        "is_manifold": is_manifold_count,
        "outlier_low_count": outlier_low_count,
        "outlier_high_count": outlier_high_count
    }

    # Save the global statistics to a CSV file
    pd.DataFrame([global_statistics]).to_csv(statistics_csv_path, index=False)
    logger.info(f"-----> Global statistics saved to {statistics_csv_path}")

    # Save the list of unique classes to a text file
    classes_list = shapes_data_df['obj_class'].unique()
    with open(classes_txt_path, 'w') as f:
        for obj_class in classes_list:
            f.write(f"{obj_class}\n")


def plot_distribution(df, title, plots_dir, bins=30):
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
    
    # Construct the full file path
    filename = f"{title}.png"
    save_plot_path = os.path.join(plots_dir, filename)
    
    # Save the plot using the full path
    plt.savefig(save_plot_path)
    
    # Optional: close the plot to free memory
    plt.close()


def clean_mesh(mesh):
    """ 
    Clean the mesh by removing duplicate faces and vertices, repairing holes, and ensuring consistent orientation.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh to clean.
        
    Returns:
        trimesh.Trimesh: The cleaned mesh object.
        None: If the mesh cannot be cleaned or an error occurs.
    """
    
    try:
        mesh_copy = deepcopy(mesh)
        mesh_copy.remove_degenerate_faces()          # Remove degenerate (zero-area) faces
        mesh_copy.merge_vertices()                   # Merge vertices (remove duplicated vertices)
        mesh_copy.remove_unreferenced_vertices()     # Remove vertices not referenced by any faces
        mesh_copy.remove_infinite_values()           # Remove vertices with infinite/NaN coordinates
        mesh_copy.fill_holes()                       # Fill in missing faces to close holes
        mesh_copy.fix_normals()                      # Fix any incorrect face or vertex normals

        if len(mesh_copy.faces) == 0:
            logger.warning("Mesh has no faces left after cleaning.")
            return None

        return mesh_copy

    except Exception as e:
        logger.error(f"Error during mesh cleaning: {e}")
        return None


def clean_single_mesh(row, cleaned_meshes_root):
    """ 
    Process a single mesh, clean it, save the cleaned mesh, and update its properties.
    
    Parameters:
        row (pd.Series): A row from the DataFrame containing file paths and metadata.
        cleaned_meshes_root (str): Root directory where the cleaned meshes will be saved.
        
    Returns:
        pd.Series: Updated row with cleaned mesh data.
        None: If an error occurs or the mesh cannot be cleaned.
    """
    
    obj_class = row['obj_class']
    file_name = row['file_name']
    file_path = row['file_path']
    cleaned_file_path = os.path.join(cleaned_meshes_root, obj_class, file_name)
    
    try:
        # Load the mesh using Trimesh
        mesh = trimesh.load(file_path)
        
        # Clean the mesh
        cleaned_mesh = clean_mesh(mesh)
        
        if cleaned_mesh is None:
            return None
        
        # Ensure the directory for cleaned file exists
        os.makedirs(os.path.dirname(cleaned_file_path), exist_ok=True)
        
        # Save the cleaned mesh
        cleaned_mesh.export(cleaned_file_path)
        
        return cleaned_file_path

    except Exception as e:
        logger.error(f"Error cleaning mesh {file_path}: {str(e)}")
        return None



def clean_dataset_parallel(df, cleaned_meshes_root, num_processes=16):
    """
    Parallelizes the cleaning process of the dataset.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing information about each mesh to process.
        num_processes (int): The number of processes to run in parallel.

    Returns:
        pd.DataFrame: DataFrame of valid cleaned meshes.
    """
    with mp.Pool(processes=num_processes) as pool:
        # Parallelize the cleaning of each mesh row
        results = pool.starmap(clean_single_mesh, [(row, cleaned_meshes_root) for _, row in df.iterrows()])

    # Filter out any None values (i.e., failed meshes)
    successful_paths = [res for res in results if res is not None]
    
    return successful_paths


def adjust_mesh_complexity(mesh, max_iterations=10):
    """
    Adjust mesh complexity by refining low-poly meshes and simplifying high-poly meshes.

    Parameters:
        mesh (trimesh.Trimesh): The mesh object to adjust.
        max_iterations (int): Maximum number of iterations allowed.
    Returns:
        trimesh.Trimesh: The adjusted and cleaned mesh object.
    """
    current_vertices = len(mesh.vertices)
    for _ in range(max_iterations):
        current_vertices = len(mesh.vertices)
        if current_vertices < LOWER_BOUND:
            mesh = mesh.subdivide()
        elif current_vertices > UPPER_BOUND:
            mesh = mesh.simplify_quadric_decimation(percent=0.3)
        else:
            break

    return mesh


def process_single_mesh_resample(row, resampled_root):
    """
    Process a single mesh, resample it, and save the result.
    
    Parameters:
        row (pd.Series): A row from the DataFrame containing file paths and metadata.
        resampled_root (str): Root directory where the resampled meshes will be saved.
    Returns:
        str: The file path if successful, or None if an error occurs.
    """
    obj_class = row['obj_class']
    file_name = row['file_name']
    file_path = row['file_path']
    resampled_file_path = os.path.join(resampled_root, obj_class, file_name)
    
    try:
        # Load the mesh using Trimesh
        mesh = trimesh.load(file_path)
        
        # Adjust the mesh complexity
        resampled_mesh = adjust_mesh_complexity(mesh)
        
        # Ensure the directory for resampled file exists
        os.makedirs(os.path.dirname(resampled_file_path), exist_ok=True)
        
        # Save the resampled mesh
        resampled_mesh.export(resampled_file_path)
        
        return resampled_file_path

    except Exception as e:
        logger.error(f"Error processing mesh {file_path}: {str(e)}")
        return None


def process_meshes_parallel(df, resampled_root, num_processes=16):
    """
    Parallelize the resampling of meshes across multiple processors.
    
    Parameters:
        df (pd.DataFrame): The dataset containing file paths and metadata.
        resampled_root (str): The root directory where resampled files will be saved.
        num_processes (int): The number of parallel processes to use.
    
    Returns:
        list: List of paths to successfully resampled meshes.
    """
    with mp.Pool(processes=num_processes) as pool:
        # Parallelize the processing of each mesh row
        results = pool.starmap(process_single_mesh_resample, [(row, resampled_root) for _, row in df.iterrows()])

    # Filter out any None values (i.e., failed meshes)
    successful_paths = [res for res in results if res is not None]
    
    return successful_paths


if __name__ == "__main__":
    
    try:
        import fast_simplification
        print("fast_simplification module loaded successfully")
    except ImportError as e:
        print(f"Error loading fast_simplification: {e}")
    
    # Create directories and CSV paths 
    create_directory(OUTPUT_PATH, overwrite=OVERWRITE, logger=logger)
    create_directory(OUTPUTS_DATA_PATH, overwrite=OVERWRITE, logger=logger)
    create_directory(OUTPUTS_PLOTS_PATH, overwrite=OVERWRITE, logger=logger)
    create_directory(CLEANED_SHAPES_PATH, overwrite=OVERWRITE, logger=logger)
    create_directory(RESAMPLED_SHAPES_PATH, overwrite=OVERWRITE, logger=logger)
    create_directory(CLEANED_RESAMPLED_SHAPES_PATH, overwrite=OVERWRITE, logger=logger)

    original_csv_path = os.path.join(OUTPUT_PATH, "shapes_data.csv")
    resampled_cleaned_csv_path = os.path.join(OUTPUT_PATH, "shapes_data_resampled_cleaned.csv")
    resampled_csv_path = os.path.join(OUTPUT_PATH, "shapes_data_resampled.csv")
    cleaned_csv_path = os.path.join(OUTPUT_PATH, "shapes_data_cleaned.csv")
    resampled_cleaned_csv_path = os.path.join(OUTPUT_PATH, "shapes_data_resampled_cleaned.csv")
    global_statistics_original_csv_path = os.path.join(OUTPUTS_DATA_PATH, "global_statistics_original.csv")
    global_statistics_resampled_csv_path = os.path.join(OUTPUTS_DATA_PATH, "global_statistics_resampled.csv")
    global_statistics_resampled_cleaned_csv_path = os.path.join(OUTPUTS_DATA_PATH, "global_statistics_resampled_cleaned.csv")
    classes_txt_path = os.path.join(OUTPUTS_DATA_PATH, "classes.txt")
    times_path = os.path.join(OUTPUTS_DATA_PATH, "times.csv")

    times = {}
    start_resampling = time.time()
    # ------------------------------------------------ ANALYZE ORIGINAL SHAPES ------------------------------------------------ #
    if not os.path.exists(original_csv_path) or OVERWRITE:
        logger.info("# Step 1: Analyzing Original database and collecting statistics...")
        shapes_data = analyze_dataset(DATASET_PATH)
        shapes_data_df = pd.DataFrame(shapes_data)
        shapes_data_df.to_csv(original_csv_path, index=False)
        # logger.info(f"-----> Original shapes data saved to '{original_csv_path}'")
    else:
        shapes_data_df = pd.read_csv(original_csv_path)
        logger.info(f"# Step 1: Loaded existing Original shapes data from '{original_csv_path}'")

    plot_distribution(shapes_data_df, "Initial distribution", OUTPUTS_PLOTS_PATH)
    
    # -------------------------------------------- SAVE INITIAL GLOBAL STATISTICS --------------------------------------------- #    
    compute_and_save_statistics(shapes_data_df, global_statistics_original_csv_path, classes_txt_path)
    
    # ----------------------------------------------------- CLEAN MESHES ------------------------------------------------------ #    
    logger.info("# Step 2: Clean Original meshes...")
    shapes_data_df = pd.read_csv(original_csv_path)
        
    # Start processing the meshes
    clean_dataset_parallel(shapes_data_df, CLEANED_SHAPES_PATH, num_processes=PROCESSORS)
    
    # logger.info(f"-----> Cleaned shapes saved to '{CLEANED_SHAPES_PATH}'")

    # ----------------------------------------------- ANALYZE CLEANED SHAPES ------------------------------------------------ #
    logger.info("# Step 3: Analyzing Cleaned dataset and collecting statistics...")
    cleaned_shapes_data = analyze_dataset(CLEANED_SHAPES_PATH)
    cleaned_shapes_data_df = pd.DataFrame(cleaned_shapes_data)
    cleaned_shapes_data_df.to_csv(cleaned_csv_path, index=False)
    # logger.info(f"-----> Cleaned shapes data saved to '{resampled_csv_path}'")

    plot_distribution(cleaned_shapes_data_df, "After Cleaning", OUTPUTS_PLOTS_PATH)
    
    # ---------------------------------------------------- RESAMPLE SHAPES ---------------------------------------------------- #    
    if not os.path.exists(RESAMPLED_SHAPES_PATH) or OVERWRITE:
        logger.info("# Step 4: Resample Cleaned meshes...")
        process_meshes_parallel(cleaned_shapes_data_df, RESAMPLED_SHAPES_PATH, num_processes=PROCESSORS)
        # logger.info(f"-----> Resampled shapes saved to '{RESAMPLED_SHAPES_PATH}'")
    
    # ----------------------------------------------- ANALYZE RESAMPLED SHAPES ------------------------------------------------ #    
    if not os.path.exists(resampled_csv_path) or OVERWRITE:
        logger.info("# Step 5.1: Analyzing Resampled dataset and collecting statistics...")
        resampled_shapes_data = analyze_dataset(RESAMPLED_SHAPES_PATH)
        resampled_shapes_data_df = pd.DataFrame(resampled_shapes_data)
        resampled_shapes_data_df.to_csv(resampled_csv_path, index=False)
        # logger.info(f"-----> Resampled shapes data saved to '{resampled_csv_path}'")
    else:
        resampled_shapes_data_df = pd.read_csv(resampled_csv_path)
        logger.info(f"# Step 5.1: Loaded existing resampled shapes data from '{resampled_csv_path}'")

    plot_distribution(resampled_shapes_data_df, "After Reshaping", OUTPUTS_PLOTS_PATH)
    
    # --------------------------------------------- REMOVE SHAPES STILL OUTLIERS ---------------------------------------------- #   
    logger.info(f"# Step 5.2: Remove shapes that are still outliers from the Resampled meshes...") 
    remaining_shapes_df = resampled_shapes_data_df[(resampled_shapes_data_df['vertices'] >= LOWER_BOUND) & 
                                               (resampled_shapes_data_df['vertices'] <= UPPER_BOUND)]
    
    remaining_shapes_df.to_csv(resampled_csv_path)
    
    outliers_df = resampled_shapes_data_df[(resampled_shapes_data_df['vertices'] < LOWER_BOUND) | (resampled_shapes_data_df['vertices'] > UPPER_BOUND)]
    
    # Delete files for the outliers based on the 'path' column
    for file_path in outliers_df['file_path']:
        if os.path.exists(file_path):
            os.remove(file_path)
            # logger.info(f"Deleted file: {file_path}")
        else:
            logger.info(f"File not found: {file_path}")

    # logger.info(f"-----> Deleted {outliers_df.shape[0]} files")
    
    plot_distribution(remaining_shapes_df, "After removing outliers", OUTPUTS_PLOTS_PATH)
    
    # --------------------------------------------- SAVE RESAMPLED GLOBAL STATISTICS ---------------------------------------------- # 
    logger.info(f"# Step 6: Compute global statistics for the Resampled meshes...")    
    compute_and_save_statistics(remaining_shapes_df, global_statistics_resampled_csv_path, classes_txt_path)
    
    # ----------------------------------------------------- CLEAN RESAMPLED MESHES ------------------------------------------------------ #    
    logger.info("# Step 7: Clean Resampled meshes...")
        
    # Start processing the meshes
    clean_dataset_parallel(remaining_shapes_df, CLEANED_RESAMPLED_SHAPES_PATH, num_processes=PROCESSORS)
    
    # logger.info(f"-----> Resampled & Cleaned shapes saved to '{CLEANED_RESAMPLED_SHAPES_PATH}'")

    # ----------------------------------------------- ANALYZE CLEANED SHAPES ------------------------------------------------ #
    logger.info("# Step 8: Analyzing Resampled & Cleaned dataset and collecting statistics...")
    resampled_cleaned_shapes_data = analyze_dataset(CLEANED_RESAMPLED_SHAPES_PATH)
    resampled_cleaned_shapes_data_df = pd.DataFrame(resampled_cleaned_shapes_data)
    resampled_cleaned_shapes_data_df.to_csv(resampled_cleaned_csv_path, index=False)
    # logger.info(f"-----> Resampled & Cleaned shapes data saved to '{resampled_cleaned_csv_path}'")

    plot_distribution(resampled_cleaned_shapes_data_df, "After Resampling and Cleaning", OUTPUTS_PLOTS_PATH)
    
    # --------------------------------------------- SAVE CLEANED & RESAMPLED GLOBAL STATISTICS ---------------------------------------------- # 
    logger.info("# Step 9: Compute Global Statistics from Resampled & Cleaned shapes and collecting statistics...")   
    compute_and_save_statistics(resampled_cleaned_shapes_data_df, global_statistics_resampled_cleaned_csv_path, classes_txt_path)
    # logger.info(f"-----> Global statistics for resampled and cleaned data saved to '{global_statistics_resampled_cleaned_csv_path}'")

    end_resampling = time.time()
    
    times['resampling'] = end_resampling - start_resampling
    times_df = pd.DataFrame([times])
    times_df.to_csv(times_path, index=False)


















