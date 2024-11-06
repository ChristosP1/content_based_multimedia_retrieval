import numpy as np
import trimesh
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
import time
import random
import multiprocessing as mp
import numpy as np


def sample_points(mesh, sample_size, points_per_sample, compute_func, **kwargs):
    """
    Samples random points on the mesh and applies the compute function to them.
    
    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        sample_size (int): The number of samples to draw.
        points_per_sample (int): Number of points required for each sample (e.g., 2 for distances, 3 for triangles).
        compute_func (function): The function to apply to the points.
        kwargs: Additional arguments to pass to the compute function.
    
    Returns:
        np.ndarray: An array of computed values.
    """
    results = []
    vertices = list(mesh.vertices)  # Get vertex positions
    for _ in range(sample_size):
        # Directly sample from vertices (similar to the student's approach)
        sampled_points = random.sample(vertices, points_per_sample)
        result = compute_func(sampled_points, **kwargs)
        results.append(result)
    return np.array(results)


# ----------------------------------------------------- A3 ----------------------------------------------------- #
def compute_angle(pts):
    v1, v2, v3 = pts

    # Vector between points (now using a common vertex as the student's code does)
    e1 = v2 - v1
    e2 = v3 - v1

    # Cosine of the angle
    cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip to avoid numerical errors
    
    # Return the angle in degrees
    return np.degrees(angle)

def compute_angles(mesh, sample_size):
    return sample_points(mesh, sample_size, 3, compute_angle)


# ----------------------------------------------------- D1 ----------------------------------------------------- #
def compute_dist_to_centroid(pts, centroid):
    v1 = pts
    return np.linalg.norm(v1 - centroid)

def compute_d1(mesh, sample_size):
    centroid = mesh.centroid  # Use mesh centroid
    return sample_points(mesh, sample_size, 1, compute_dist_to_centroid, centroid=centroid)


# ----------------------------------------------------- D2 ----------------------------------------------------- #
def compute_dist_between_vertices(pts):
    v1, v2 = pts
    return np.linalg.norm(v1 - v2)

def compute_d2(mesh, sample_size):
    return sample_points(mesh, sample_size, 2, compute_dist_between_vertices)


# ----------------------------------------------------- D3 ----------------------------------------------------- #
def compute_triangle_area(pts):
    v1, v2, v3 = pts

    # Use trimesh triangle area computation
    triangle = np.array([[v1, v2, v3]])
    return np.sqrt(trimesh.triangles.area(triangle))

def compute_d3(mesh, sample_size):
    return sample_points(mesh, sample_size, 3, compute_triangle_area)


# ----------------------------------------------------- D4 ----------------------------------------------------- #
def compute_tetrahedron_volume(pts):
    v1, v2, v3, v4 = pts

    # Compute the volume of the tetrahedron formed by the 4 points
    tetrahedron_matrix = np.array([v2 - v1, v3 - v2, v4 - v1])
    volume = np.abs(np.linalg.det(tetrahedron_matrix)) / 6.0
    return np.cbrt(volume)

def compute_d4(mesh, sample_size):
    return sample_points(mesh, sample_size, 4, compute_tetrahedron_volume)



                                    # --------------------------------- #
                                    # --------------------------------- #

             

def compute_fd_bins(data):
    """
    Compute the number of bins using the Freedman-Diaconis rule.
    
    Parameters:
        data (np.array): The array of data to calculate bins for.
        
    Returns:
        int: Number of bins.
    """
    if len(data) < 2:
        return 10  # Default bins if data is too small

    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25  # Interquartile range
    bin_width = 2 * iqr * (len(data) ** (-1/3))  # Bin width based on Freedman-Diaconis rule
    if bin_width == 0:
        return 10  # Default number of bins if bin width is 0
    return int(np.ceil((data.max() - data.min()) / bin_width))


def compute_bins_for_chunk(df_chunk, sample_size, preprocessed_dataset_path):
    """
    Compute the bins for each descriptor in a chunk of the dataset.
    
    Parameters:
        df_chunk (pd.DataFrame): Chunk of the dataset.
        sample_size (int): The number of points to sample for descriptors.
        preprocessed_dataset_path (str): Path to the preprocessed dataset.
        
    Returns:
        dict: List of bins for each descriptor (A3, D1, D2, D3, D4) for this chunk.
    """
    all_bins = {'A3': [], 'D1': [], 'D2': [], 'D3': [], 'D4': []}

    for _, row in df_chunk.iterrows():
        obj_class = row['obj_class']
        file_name = row['file_name']
        preprocessed_file_path = os.path.join(preprocessed_dataset_path, obj_class, file_name)

        # Load the mesh
        try:
            mesh = trimesh.load(preprocessed_file_path)
        except Exception as e:
            print(f"Could not load mesh: {preprocessed_file_path}, {e}")
            continue
        
        # Compute each descriptor
        a_3 = compute_angles(mesh, sample_size)
        d_1 = compute_d1(mesh, sample_size)
        d_2 = compute_d2(mesh, sample_size)
        d_3 = compute_d3(mesh, sample_size)
        d_4 = compute_d4(mesh, sample_size)

        # Calculate bins for each descriptor using Freedman-Diaconis rule
        all_bins['A3'].append(compute_fd_bins(a_3))
        all_bins['D1'].append(compute_fd_bins(d_1))
        all_bins['D2'].append(compute_fd_bins(d_2))
        all_bins['D3'].append(compute_fd_bins(d_3))
        all_bins['D4'].append(compute_fd_bins(d_4))

    return all_bins

def aggregate_bins(all_bins_list):
    """
    Aggregate the bins from multiple processes and calculate the mean for each descriptor.
    
    Parameters:
        all_bins_list (list of dicts): List of dictionaries containing bins from each process.
        
    Returns:
        dict: Average number of bins for each descriptor.
    """
    aggregated_bins = {'A3': [], 'D1': [], 'D2': [], 'D3': [], 'D4': []}

    # Flatten all bins from different chunks into one list per descriptor
    for bins in all_bins_list:
        aggregated_bins['A3'].extend(bins['A3'])
        aggregated_bins['D1'].extend(bins['D1'])
        aggregated_bins['D2'].extend(bins['D2'])
        aggregated_bins['D3'].extend(bins['D3'])
        aggregated_bins['D4'].extend(bins['D4'])

    # Calculate the mean number of bins for each descriptor
    avg_bins = {desc: int(np.mean(aggregated_bins[desc])) for desc in aggregated_bins}
    return avg_bins


def compute_all_bins_parallel(preprocessed_shapes_df, sample_size, preprocessed_dataset_path, num_processes=4):
    """
    Compute the bins for all descriptors (A3, D1, D2, D3, D4) across all meshes in parallel.
    
    Parameters:
        preprocessed_shapes_df (pd.DataFrame): DataFrame of the dataset containing file paths.
        sample_size (int): Number of points to sample for each descriptor.
        preprocessed_dataset_path (str): Path to the preprocessed dataset.
        num_processes (int): Number of processes to run in parallel.
        
    Returns:
        dict: Average number of bins for each descriptor.
    """
    # Split the dataframe into chunks, one for each process
    df_chunks = np.array_split(preprocessed_shapes_df, num_processes)

    # Use multiprocessing to compute bins in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(compute_bins_for_chunk, 
                               [(df_chunk, sample_size, preprocessed_dataset_path) for df_chunk in df_chunks])

    # Aggregate the bins from all processes
    avg_bins = aggregate_bins(results)
    
    pd.DataFrame([avg_bins]).to_csv('outputs/data/average_bins_local.csv', index=False)

    return avg_bins

                      
def compute_local_descriptors(mesh, sample_size, avg_bins, random_seed=42):
    """
    Computes the local descriptors and returns a histogram for each descriptor.
    """
    np.random.seed(random_seed)
    
    hist_ranges = {
        'A3': (0, 180),  # Degrees for angles
        'D1': (0, np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])),  # Bounding box diagonal for distances
        'D2': (0, np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])),  # Bounding box diagonal for distances
        'D3': (0, np.sqrt(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) ** 2)),  
        'D4': (0, np.cbrt(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) ** 3)), 
    }

    # Compute each descriptor
    a_3 = compute_angles(mesh, sample_size)
    d_1 = compute_d1(mesh, sample_size)
    d_2 = compute_d2(mesh, sample_size)
    d_3 = compute_d3(mesh, sample_size)
    d_4 = compute_d4(mesh, sample_size)

    # Normalize histograms (they sum to 1)
    def normalized_histogram(values, bins, range):
        hist = np.histogram(values, bins=bins, range=range)
        return hist[0] / np.sum(hist[0]) if np.sum(hist[0]) > 0 else hist[0]

    # Create normalized histograms for each descriptor
    local_features = {
        'A3': list(normalized_histogram(a_3,int(avg_bins['A3'].values[0]), range=hist_ranges['A3'])),
        'D1': list(normalized_histogram(d_1, int(avg_bins['D1'].values[0]), range=hist_ranges['D1'])),
        'D2': list(normalized_histogram(d_2, int(avg_bins['D2'].values[0]), range=hist_ranges['D2'])),
        'D3': list(normalized_histogram(d_3, int(avg_bins['D3'].values[0]), range=hist_ranges['D3'])),
        'D4': list(normalized_histogram(d_4, int(avg_bins['D4'].values[0]), range=hist_ranges['D4']))
    }
    return pd.Series(local_features)


def compute_local_descriptors_parallel(df_chunk, avg_bins, sample_size, preprocessed_dataset_path):
    local_descriptors = []

    for _, row in df_chunk.iterrows():
        obj_class = row['obj_class']
        file_name = row['file_name']
        preprocessed_file_path = os.path.join(preprocessed_dataset_path, obj_class, file_name)

        # Load the mesh
        try:
            mesh = trimesh.load(preprocessed_file_path)
        except Exception as e:
            print(f"Could not load mesh: {preprocessed_file_path}, {e}")
            continue

        # Compute local descriptors for the mesh
        descriptors = compute_local_descriptors(mesh, sample_size, avg_bins)

        # Add object class and file name for reference
        descriptors['obj_class'] = obj_class
        descriptors['file_name'] = file_name

        local_descriptors.append(descriptors)

    return local_descriptors


def run_in_parallel(preprocessed_shapes_df, avg_bins, sample_size, preprocessed_dataset_path, num_processes=4):
    """
    Run the local descriptor computations in parallel using multiprocessing.

    Parameters:
        preprocessed_shapes_df (pd.DataFrame): The dataframe containing the preprocessed shapes.
        avg_bins (dict): The average number of bins for each descriptor.
        sample_size (int): The number of random points to sample from the mesh.
        preprocessed_dataset_path (str): Path to the dataset containing the preprocessed meshes.
        num_processes (int): Number of parallel processes to run.
        
    Returns:
        pd.DataFrame: A DataFrame containing all the computed local descriptors.
    """
    # Split the dataframe into chunks, one for each process
    df_chunks = np.array_split(preprocessed_shapes_df, num_processes)

    # Use multiprocessing to compute descriptors in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(compute_local_descriptors_parallel, 
                               [(df_chunk, avg_bins, sample_size, preprocessed_dataset_path) for df_chunk in df_chunks])

    # Flatten the list of results (since each process returns a list of descriptors)
    flattened_results = [item for sublist in results for item in sublist]
    
    # Convert the list of descriptors into a DataFrame
    local_descriptors_df = pd.DataFrame(flattened_results)
    
    return local_descriptors_df




if __name__ == "__main__":
    """Main function to load data, compute descriptors, and save results."""
    
    PREPROCESSED_DATASET_PATH = 'datasets/dataset_snippet_medium_normalized'
    OUTPUTS_DATA_PATH = 'outputs/data' 
    preprocessed_dataset_csv_path = 'outputs/shapes_data_normalized.csv'
    
    # Load original and preprocessed shape metadata
    preprocessed_shapes_df = pd.read_csv(preprocessed_dataset_csv_path)
    average_bins_path = 'outputs/data/average_bins_local.csv'

    times_path = os.path.join(OUTPUTS_DATA_PATH, "times.csv")
    times_df = pd.read_csv(os.path.join(OUTPUTS_DATA_PATH, "times.csv"))
    start_local_descriptors = time.time()
    
    sample_size = 4000
    num_processes = 10
    
    # First, calculate the average number of bins for each descriptor
    if not os.path.exists(average_bins_path):
        avg_bins = compute_all_bins_parallel(preprocessed_shapes_df, sample_size=sample_size, preprocessed_dataset_path=PREPROCESSED_DATASET_PATH, num_processes=num_processes)
        print(avg_bins)
    else:
        print("Average bins CSV exists!")
        avg_bins = pd.read_csv(average_bins_path)
    
    local_descriptors = []
    
    print("Start procesing")
    # Run the local descriptor computation in parallel
    local_descriptors_df = run_in_parallel(preprocessed_shapes_df, avg_bins, sample_size, PREPROCESSED_DATASET_PATH, num_processes=num_processes)
        
    # Save the final descriptors to a CSV file
    local_descriptors_df.to_csv(os.path.join(OUTPUTS_DATA_PATH, 'local_descriptors.csv'), index=False)
    
    end_local_descriptors = time.time()
    
    times_df['local_desc'] = end_local_descriptors - start_local_descriptors
    
    times_df.to_csv(times_path, index=False)
