import scipy as sp
import numpy as np
import pandas as pd
import time


def compute_knn_precision(dataset_path, output_csv, k=10):
    """
    Computes the Precision@10 for all objects in the dataset using KDTree.

    Parameters:
        dataset_path (str): Path to the dataset CSV file with combined descriptors.
        output_csv (str): Path to save the output CSV file with precision@10 and retrieval times.
        k (int): Number of nearest neighbors to retrieve (default is 10).
    """
    # Load the dataset
    dataset_features_df = pd.read_csv(dataset_path)

    # Prepare KDTree input by removing 'file_name' and 'obj_class' columns
    kd_tree_features_df = dataset_features_df.drop(columns=['file_name', 'obj_class'])

    # Build KDTree
    print("Building KDTree... ", end="")
    kdtree = sp.spatial.KDTree(kd_tree_features_df)
    print("Finished.")

    # Initialize results list
    results = []

    # Loop through each object in the dataset
    for idx, row in dataset_features_df.iterrows():
        query_features = row.drop(['file_name', 'obj_class']).values.reshape(1, -1)
        query_class = row['obj_class']
        query_name = row['file_name']

        # Measure retrieval time
        start_time = time.time()
        knn_distances, knn_indices = kdtree.query(query_features, k=k)
        retrieval_time = time.time() - start_time

        # Compute Precision@10
        knn_classes = dataset_features_df.iloc[knn_indices[0]]['obj_class']
        precision_10 = sum(knn_classes == query_class) / k

        # Append results
        results.append({
            'file_name': query_name,
            'obj_class': query_class,
            'precision_10': precision_10,
            'retrieval_time': retrieval_time
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    dataset_path = "outputs/data/combined_descriptors.csv"
    output_csv = "outputs/eval/knn_precision_results.csv"
    compute_knn_precision(dataset_path, output_csv)

