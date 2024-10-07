import pymeshlab
import os
import logging
import pandas as pd
import math
import multiprocessing as mp
import numpy as np

from prep_resampling import analyze_dataset, create_directory, plot_distribution


OVERWRITE = True
# Number of parallel processors
PROCESSORS = 10

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()


# REMESHED_SHAPES_PATH = 'datasets/dataset_snippet_small_remeshed'
REMESHED_SHAPES_PATH = 'datasets/dataset_snippet_medium_remeshed'

OUTPUT_PATH = 'outputs'
OUTPUTS_DATA_PATH = 'outputs/data'
OUTPUTS_PLOTS_PATH = 'outputs/plots'


# def adjust_mesh_complexity(mesh, lower_bound=4000, upper_bound=6000, max_iterations=5):
#     """
#     Refine low-poly meshes and simplify high-poly meshes to stay within a given vertex range.

#     Parameters:
#         mesh (trimesh.Trimesh): The mesh object to adjust.
#         lower_bound (int): The lower vertex limit (default: 4000).
#         upper_bound (int): The upper vertex limit (default: 6000).
#         max_iterations (int): Maximum number of refinement iterations allowed.

#     Returns:
#         trimesh.Trimesh: The adjusted mesh object.
#     """
#     current_vertices = mesh.current_mesh().vertex_number()

#     for _ in range(max_iterations):
#         current_vertices = mesh.current_mesh().vertex_number()

#         if current_vertices < lower_bound:
#             # If the vertex count is too low, subdivide the mesh
#             mesh.apply_filter('meshing_surface_subdivision_catmull_clark')
#             print(f"Subdividing mesh to increase vertex count: {current_vertices} -> {mesh.current_mesh().vertex_number()}")

#         elif current_vertices > upper_bound:
#             # If the vertex count is too high, simplify the mesh
#             mesh.apply_filter('meshing_decimation_quadric_edge_collapse',
#             targetperc=0.9,
#             )
#             print(f"Simplifying mesh to reduce vertex count: {current_vertices} -> {mesh.current_mesh().vertex_number()}")

#         else:
#             break

#     return mesh

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
        
        
        # try:
        #     # Check the final vertex count
        #     vertex_count = ms.current_mesh().vertex_number()

        #     # If the vertex count is out of the target range, apply refinement
        #     if vertex_count < 4000 or vertex_count > 6000:
        #         print(f"Refining mesh with vertex count {vertex_count}")
        #         adjust_mesh_complexity(ms, lower_bound=4000, upper_bound=6000)
        #         return ms.current_mesh()

        # except Exception as e:
        #     print(f"An error occurred during remeshing and refinement: {e}")
        #     return None
        

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
            logger.info(f"Saved remeshed file to: {remeshed_file_path}")

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

    logger.info("Start the processes")
    # Use multiprocessing to remesh in parallel
    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(apply_remeshing, [(chunk, remeshed_root) for chunk in df_chunks])



if __name__ == "__main__":
    
    create_directory(REMESHED_SHAPES_PATH, overwrite=OVERWRITE, logger=logger)
    remeshed_csv_path = os.path.join(OUTPUT_PATH, "shapes_data_remeshed.csv")
    
    if not os.path.exists(remeshed_csv_path) or OVERWRITE:
        logger.info("# Step 5. # Remesh shapes...")
                
        resampled_shapes_df = pd.read_csv('outputs/shapes_data_resampled.csv')
        
        parallel_remeshing(resampled_shapes_df, REMESHED_SHAPES_PATH, num_processes=PROCESSORS)       
        
        remeshed_data = analyze_dataset(REMESHED_SHAPES_PATH)
        remeshed_shapes_df = pd.DataFrame(remeshed_data)
        
        remeshed_shapes_df.to_csv(remeshed_csv_path, index=False)
        logger.info(f"Remeshed shapes data saved to '{remeshed_csv_path}'")
        
        plot_distribution(remeshed_shapes_df, "Remeshed distribution", OUTPUTS_PLOTS_PATH)