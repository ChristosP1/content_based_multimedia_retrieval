import numpy as np
import trimesh
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')
import logging
import math
import os
import random


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
    for _ in range(sample_size):
        sampled_points = np.random.choice(len(mesh.vertices), points_per_sample, replace=False)
        result = compute_func(mesh, sampled_points, **kwargs)
        results.append(result)
    return np.array(results)


# ----------------------------------------------------- A3 ----------------------------------------------------- #
def compute_angle(mesh, pts):
    verts = mesh.vertices
    v1 = verts[pts[0]]
    v2 = verts[pts[1]]
    v3 = verts[pts[2]]

    # Vector between points
    e = v1 - v2
    f = v2 - v3

    # Cosine of the angle
    cos_angle = np.dot(e, f) / (np.linalg.norm(e) * np.linalg.norm(f))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip to avoid numerical errors
    
    # Return the angle in degrees
    return np.degrees(angle)

def compute_angles(mesh, sample_size):
    return sample_points(mesh, sample_size, 3, compute_angle)


# ----------------------------------------------------- D1 ----------------------------------------------------- #
def compute_dist_to_centroid(mesh, pts, centroid):
    verts = mesh.vertices
    v1 = verts[pts[0]]
    return np.linalg.norm(v1 - centroid)

def compute_d1(mesh, sample_size):
    centroid = mesh.centroid  # Use mesh centroid
    return sample_points(mesh, sample_size, 1, compute_dist_to_centroid, centroid=centroid)


# ----------------------------------------------------- D2 ----------------------------------------------------- #
def compute_dist_between_vertices(mesh, pts):
    verts = mesh.vertices
    v1 = verts[pts[0]]
    v2 = verts[pts[1]]
    return np.linalg.norm(v1 - v2)

def compute_d2(mesh, sample_size):
    return sample_points(mesh, sample_size, 2, compute_dist_between_vertices)


# ----------------------------------------------------- D3 ----------------------------------------------------- #
def compute_triangle_area(mesh, pts):
    verts = mesh.vertices
    v1 = verts[pts[0]]
    v2 = verts[pts[1]]
    v3 = verts[pts[2]]

    # Use trimesh triangle area computation
    triangle = np.array([[v1, v2, v3]])
    return np.sqrt(trimesh.triangles.area(triangle))

def compute_d3(mesh, sample_size):
    return sample_points(mesh, sample_size, 3, compute_triangle_area)


# ----------------------------------------------------- D4 ----------------------------------------------------- #
def compute_tetrahedron_volume(mesh, pts):
    verts = mesh.vertices
    v1 = verts[pts[0]]
    v2 = verts[pts[1]]
    v3 = verts[pts[2]]
    v4 = verts[pts[3]]

    # Compute the volume of the tetrahedron formed by the 4 points
    tetrahedron_matrix = np.array([v2 - v1, v3 - v1, v4 - v1])
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
        return 10  # default bins if data is too small

    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25  # Interquartile range
    bin_width = 2 * iqr * (len(data) ** (-1/3))  # Bin width based on Freedman-Diaconis rule
    if bin_width == 0:
        return 10  # default number of bins if bin width is 0
    return int(np.ceil((data.max() - data.min()) / bin_width))
                      
                                                
def compute_local_descriptors(mesh, sample_size):
    """
    Computes the local descriptors and returns a histogram for each descriptor.
    """
    
    hist_ranges = {
        # The possible angles between vectors range from 0 to 180 degrees (in 3D)
        'A3': (0, 180),  # Degrees for angles
        
        # The maximum possible distance between two points in a shape is the diagonal of the bounding box enclosing the shape. 
        # Therefore, the range is from 0 to the length of the bounding box diagonal
        'D1': (0, np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])),  
        'D2': (0, np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])),
        
        # The maximum possible triangle area is based on the maximum distances between vertices. 
        # This could theoretically be the square root of the bounding box diagonal's square
        'D3': (0, np.sqrt(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) ** 2)),  
        
        # The maximum volume of a tetrahedron is based on the maximum distances between four vertices. 
        # The cube root of the bounding box diagonal's cube is a reasonable upper bound
        'D4': (0, np.cbrt(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) ** 3)),  
    }

    # Compute each descriptor
    a_3 = compute_angles(mesh, sample_size)
    d_1 = compute_d1(mesh, sample_size)
    d_2 = compute_d2(mesh, sample_size)
    d_3 = compute_d3(mesh, sample_size)
    d_4 = compute_d4(mesh, sample_size)
    
    # Determine number of bins for histograms based on D1 distances
    nr_bins = compute_fd_bins(d_1)
    # nr_bins = 10

    # Return histograms for each descriptor
    local_features = {
        'A3': np.histogram(a_3, nr_bins, range=hist_ranges['A3'])[0],
        'D1': np.histogram(d_1, nr_bins, range=hist_ranges['D1'])[0],
        'D2': np.histogram(d_2, nr_bins, range=hist_ranges['D2'])[0],
        'D3': np.histogram(d_3, nr_bins, range=hist_ranges['D3'])[0],
        'D4': np.histogram(d_4, nr_bins, range=hist_ranges['D4'])[0]
    }

    return pd.Series(local_features)



if __name__ == "__main__":
    """Main function to load data, compute descriptors, and save results."""
    
    PREPROCESSED_DATASET_PATH = 'datasets/dataset_snippet_medium_normalized'
    OUTPUTS_DATA_PATH = 'outputs/data'
    
    preprocessed_dataset_csv_path = 'outputs/shapes_data_normalized.csv'
    
    # Load original and preprocessed shape metadata
    preprocessed_shapes_df = pd.read_csv(preprocessed_dataset_csv_path)
    
    local_descriptors = []
    
    print("Start procesing")
    # Loop through each shape in the dataset and compute descriptors
    for _, row in preprocessed_shapes_df.iterrows():
        obj_class = row['obj_class']
        file_name = row['file_name']
        
        preprocessed_file_path = os.path.join(PREPROCESSED_DATASET_PATH, obj_class, file_name)
        
        # Load the mesh
        try:
            mesh = trimesh.load(preprocessed_file_path)
        except Exception as e:
            print(f"Could not load mesh: {preprocessed_file_path}, {e}")
            continue
        
        sample_size = 2000  # Define a sample size for random vertices

        # Compute local descriptors for the mesh
        descriptors = compute_local_descriptors(mesh, sample_size)
        
        # Add object class and file name for reference
        descriptors['obj_class'] = obj_class
        descriptors['file_name'] = file_name
        
        local_descriptors.append(descriptors)
    
    # Convert list of dictionaries into a DataFrame
    local_descriptors_df = pd.DataFrame(local_descriptors)
    
    # Save the final descriptors to a CSV file
    local_descriptors_df.to_csv(os.path.join(OUTPUTS_DATA_PATH, 'local_descriptors.csv'), index=False)
    
    # Print the first few rows for verification
    print(local_descriptors_df.head())

