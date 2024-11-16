import pymeshlab
import os
import logging
import pandas as pd
import math
import multiprocessing as mp
import numpy as np
import time

from prep_resampling import analyze_dataset, create_directory, plot_distribution, clean_dataset_parallel, compute_and_save_statistics


OVERWRITE = True
# Number of parallel processors
PROCESSORS = 10

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()


# REMESHED_SHAPES_PATH = 'datasets/dataset_snippet_small_remeshed'
REMESHED_SHAPES_PATH = 'datasets/dataset_original_remeshed'
CLEANED_REMESHED_SHAPES_PATH = 'datasets/dataset_original_remeshed_cleaned'

OUTPUT_PATH = 'outputs'
OUTPUTS_DATA_PATH = 'outputs/data'
OUTPUTS_PLOTS_PATH = 'outputs/plots'


def isotropic_remesh(ms, target_vertex_count=5000, initial_scaling_factor=0.8, min_scaling_factor=0.1, decrement=0.1):
    """
    Perform isotropic remeshing on a mesh to achieve a target face count.
    
    Parameters:
        ms (pymeshlab.Mesh): The input mesh.
        target_vertex_count (int): Desired number of vertices.
        initial_scaling_factor (float): Initial scaling factor for calculating target edge length.
        min_scaling_factor (float): Minimum scaling factor to stop the iteration.
        decrement (float): Amount by which the scaling factor is reduced in each iteration.
    
    Returns:
        pymeshlab.Mesh: The remeshed mesh, or None if an error occurs.
    """
        
    faces = ms.current_mesh().face_number()

    try:
        # Compute the mesh's surface area
        area = ms.apply_filter('get_geometric_measures')['surface_area']
        scaling_factor = initial_scaling_factor
                
        # Iteratively adjust scaling factor until the vertex count is just above the target
        while scaling_factor >= min_scaling_factor:
            # Calculate target edge length based on the current scaling factor and surface area
            targetlen = scaling_factor * math.sqrt((4 * area) / (target_vertex_count * math.sqrt(3)))
            
            
            # Step 1: Perform isotropic remeshing with calculated targetlen
            ms.apply_filter('meshing_isotropic_explicit_remeshing',
                targetlen=pymeshlab.PureValue(targetlen),
                iterations=5,
            )

            # Get the current number of vertices after remeshing
            vertex_count = ms.current_mesh().vertex_number()
            
            # print(f"Isotropic remeshing applied with scaling factor {scaling_factor}. Target length {targetlen} Vertex count: {vertex_count}")
            
            # If the vertex count is close to but above the target, stop the iteration
            if vertex_count >= target_vertex_count:
                break
            
            # Decrease the scaling factor for the next iteration
            scaling_factor = scaling_factor - decrement
        
        # Step 2: Simplify the mesh to the target face count (optional, based on original face count)
        ms.apply_filter('meshing_decimation_quadric_edge_collapse',
            targetfacenum=faces,
        )

        remeshed_mesh = ms.current_mesh()
        return remeshed_mesh

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def apply_remeshing(df_chunk, remeshed_root):
    """ 
    Process a chunk of meshes from the DataFrame, apply isotropic remeshing, and save the results.
    
    Parameters:
        df_chunk (pd.DataFrame): A chunk of the dataset containing file paths and metadata for the meshes.
        remeshed_root (str): The root directory where the remeshed files will be saved.
    
    Returns:
        None: Saves remeshed files to the specified directory.
    """
    
    for _, row in df_chunk.iterrows():
        obj_class = row['obj_class']
        file_name = row['file_name']
        file_path = row['file_path']
        remeshed_file_path = os.path.join(remeshed_root, obj_class, file_name)

        try:
            # logger.info(f"Processing: {file_path}")

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(file_path)

            # Apply isotropic remeshing
            isotropic_remesh(ms)

            # Ensure the directory for resampled file exists
            os.makedirs(os.path.dirname(remeshed_file_path), exist_ok=True)

            # Save the resampled mesh
            ms.save_current_mesh(remeshed_file_path)
            # logger.info(f"Saved remeshed file to: {remeshed_file_path}")

        except Exception as e:
            logger.error(f"Error processing mesh {file_path}: {str(e)}")


def parallel_remeshing(df, remeshed_root, num_processes=8):
    """
    Parallelize the remeshing process across multiple processors.
    
    Parameters:
        df (pd.DataFrame): The dataset containing file paths and metadata.
        remeshed_root (str): The root directory where remeshed files will be saved.
        num_processes (int): The number of parallel processes to use.
    """
    # logger.info("Split the df")
    # Split the dataframe into chunks, one for each process
    df_chunks = np.array_split(df, num_processes)

    # logger.info("Start the processes")
    # Use multiprocessing to remesh in parallel
    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(apply_remeshing, [(chunk, remeshed_root) for chunk in df_chunks])



if __name__ == "__main__":
    
    create_directory(REMESHED_SHAPES_PATH, overwrite=OVERWRITE, logger=logger)
    remeshed_data_path = os.path.join(OUTPUT_PATH, "shapes_data_remeshed.csv")
    remeshed_cleaned_data_path = os.path.join(OUTPUT_PATH, "shapes_data_remeshed_cleaned.csv")
    global_statistics_remeshed_cleaned_csv_path = os.path.join(OUTPUTS_DATA_PATH, "global_statistics_remeshed_cleaned.csv")
    classes_txt_path = os.path.join(OUTPUTS_DATA_PATH, "classes.txt")
    times_path = os.path.join(OUTPUTS_DATA_PATH, "times.csv")
    
    times_df = pd.read_csv(os.path.join(OUTPUTS_DATA_PATH, "times.csv"))
    
    start_remeshing = time.time()
    # ----------------------------------------------- REMESH CLEANED SHAPES ------------------------------------------------ #
    if not os.path.exists(remeshed_data_path) or OVERWRITE:
        logger.info("# Step 10: Remesh shapes...")
                
        resampled_cleaned_shapes_df = pd.read_csv('outputs/shapes_data_resampled_cleaned.csv')
        
        parallel_remeshing(resampled_cleaned_shapes_df, REMESHED_SHAPES_PATH, num_processes=PROCESSORS)       
        
        remeshed_data = analyze_dataset(REMESHED_SHAPES_PATH)
        remeshed_shapes_df = pd.DataFrame(remeshed_data)
        
        remeshed_shapes_df.to_csv(remeshed_data_path, index=False)
        # logger.info(f"-----> Remeshed shapes data saved to '{remeshed_data_path}'")
        
        plot_distribution(remeshed_shapes_df, "Remeshed distribution", OUTPUTS_PLOTS_PATH)
    
    # ----------------------------------------------------- CLEAN RESAMPLED MESHES ------------------------------------------------------ #    
    logger.info("# Step 11: Clean remeshed meshes...")
        
    clean_dataset_parallel(remeshed_shapes_df, CLEANED_REMESHED_SHAPES_PATH, num_processes=PROCESSORS)
    
    # logger.info(f"-----> Remeshed & Cleaned shapes saved to '{CLEANED_REMESHED_SHAPES_PATH}'")

    # ----------------------------------------------- ANALYZE CLEANED SHAPES ------------------------------------------------ #
    logger.info("# Step 12: Analyzing Remeshed & Cleaned dataset and collecting statistics...")
    remeshed_cleaned_shapes_data = analyze_dataset(CLEANED_REMESHED_SHAPES_PATH)
    remeshed_cleaned_shapes_data_df = pd.DataFrame(remeshed_cleaned_shapes_data)
    remeshed_cleaned_shapes_data_df.to_csv(remeshed_cleaned_data_path, index=False)
    # logger.info(f"-----> Resampled & Cleaned shapes data saved to '{remeshed_cleaned_data_path}'")

    plot_distribution(remeshed_cleaned_shapes_data_df, "After Remeshing and Cleaning", OUTPUTS_PLOTS_PATH)
    
    # --------------------------------------------- SAVE CLEANED & RESAMPLED GLOBAL STATISTICS ---------------------------------------------- # 
    logger.info("# Step 13: Compute Global Statistics from Remeshed & Cleaned shapes and collecting statistics...")   
    compute_and_save_statistics(remeshed_cleaned_shapes_data_df, global_statistics_remeshed_cleaned_csv_path, classes_txt_path)
    # logger.info(f"-----> Global statistics for remeshed and cleaned data saved to '{global_statistics_remeshed_cleaned_csv_path}'")
    
    end_remeshing = time.time()
    
    times_df['remeshing'] = end_remeshing - start_remeshing
    
    times_df.to_csv(times_path, index=False)