from collections import defaultdict
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean
from multiprocessing import Pool

print("Imports successful.")


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


def process_single_object(args):
    """Process a single object in the dataset to find Precision and Recall."""
    row, class_counts, total_items  = args
    input_desc_df = row.to_frame().transpose()

    # Regular search
    distances = compute_distances(input_desc_df)
    similar_meshes = standardize_and_save_similarity_scores(distances)

    # Adjust number of retrieved objects to match the class count
    num_retrieved = class_counts[row['obj_class']]
    top_retrieved = similar_meshes.head(num_retrieved)
    top_10_retrieved = similar_meshes.head(10)

    # Calculate precision and recall
    input_class = row['obj_class']
    num_same_class = top_retrieved[top_retrieved['obj_class'] == input_class].shape[0]
    same_class_top_10 = top_10_retrieved[top_10_retrieved['obj_class'] == input_class].shape[0]
    num_different_class = top_retrieved[top_retrieved['obj_class'] != input_class].shape[0]
    
    TP = num_same_class
    TP_10 = num_same_class
    
    FP = num_different_class
    
    FN = class_counts[input_class] - TP
    FN_10 = class_counts[input_class] - TP_10
    
    TN = total_items - (TP + FP + FN)
    TN_10 = total_items - (TP_10 + FP + FN_10)
    
    precision = num_same_class / num_retrieved if num_retrieved > 0 else 0  
    precision_10 = same_class_top_10 / 10  
    recall = num_same_class / class_counts[input_class] if class_counts[input_class] > 0 else 0
    
    # Calculate accuracy
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    accuracy_10 = (TP_10 + TN_10) / (TP_10 + FP + FN_10 + TN_10) if (TP_10 + FP + FN_10 + TN_10) > 0 else 0

    return row['file_name'], input_class, precision, precision_10, recall, accuracy, accuracy_10


def evaluate_retrieval_engine(descriptors_file, output_csv):
    # Load the dataset descriptors
    dataset_df = pd.read_csv(descriptors_file)
    
    # Count the number of objects per class in the dataset
    class_counts = dataset_df['obj_class'].value_counts().to_dict()
    all_items = len(dataset_df.index)
    
    # Use multiprocessing to process each object in the dataset
    with Pool() as pool:
        results = pool.map(
            process_single_object,
            [(row, class_counts, all_items) for _, row in dataset_df.iterrows()]
        )
    
    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['file_name', 'obj_class', 'precision', 'precision_10', 'recall', 'accuracy', 'accuracy_10']).round(2)
    results_df.to_csv(output_csv, index=False)
    
    print(f"Evaluation results saved to {output_csv}")


if __name__ == "__main__":
    descriptors_file = "outputs/data/combined_descriptors.csv"
    output_csv = "outputs/eval/evaluation_results.csv"
    evaluate_retrieval_engine(descriptors_file, output_csv)
