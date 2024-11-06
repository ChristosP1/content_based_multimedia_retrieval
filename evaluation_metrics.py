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


def get_dominant_class_items(similar_meshes_df, dominant_class_name):
    """
    Filter and return all items of the dominant class from the similar meshes dataframe.
    
    Parameters:
        similar_meshes_df (pd.DataFrame): DataFrame containing similar meshes with
                                          'obj_class' as one of the columns.
        dominant_class_name (str): Name of the dominant class to filter on.
    
    Returns:
        pd.DataFrame: A DataFrame containing only items from the dominant class.
    """
    # Filter the dataframe for rows matching the dominant class
    class_items_df = similar_meshes_df[similar_meshes_df['obj_class'] == dominant_class_name]
    
    return class_items_df



def find_dominant_class(standardized_distances, top_n=10):
    """
    Compute the most dominant class from the top N retrieved objects by summing the final
    scores of each class and selecting the one with the lowest cumulative score.
    
    Parameters:
        standardized_distances (pd.DataFrame): DataFrame with standardized distances, 
                                               including 'file_name', 'obj_class', 
                                               and 'final_score' columns.
        top_n (int): Number of top objects to consider (default is 10).
        
    Returns:
        str: The most dominant class.
    """
    # Select the top N objects based on final score
    top_results = standardized_distances.head(top_n)
    
    # Group by class and sum the final scores within each class
    class_scores = top_results.groupby('obj_class')['final_score'].sum()
    print(class_scores)
    sorted_scores = class_scores.sort_values()
    
    most_dominant_class = sorted_scores.index[0]
    second_dominant_class = sorted_scores.index[1] if len(sorted_scores) > 1 else None
    
    dominant_class_score = sorted_scores.iloc[0]
    second_class_score = sorted_scores.iloc[1] if second_dominant_class else float('inf')
    print(dominant_class_score)
    print(second_class_score)
    
     # Check if the most dominant class has at least twice the "dominance" (lower score) compared to the second
    if dominant_class_score <= 1.5 * second_class_score:
        print(dominant_class_score, " < ", 1.5 * second_class_score)
        return most_dominant_class
    else:
        return False


def process_single_object(args):
    """Process a single object in the dataset for both regular and enhanced recall."""
    row, class_counts = args
    input_desc_df = row.to_frame().transpose()

    # Regular search
    distances = compute_distances(input_desc_df)
    similar_meshes = standardize_and_save_similarity_scores(distances)
    top_10_regular = similar_meshes.head(10)

    # Calculate regular recall
    input_class = row['obj_class']
    num_same_class_regular = top_10_regular[top_10_regular['obj_class'] == input_class].shape[0]
    recall_regular = num_same_class_regular / class_counts[input_class] if class_counts[input_class] > 0 else 0

    # Enhanced search
    dominant_class = find_dominant_class(similar_meshes)
    if dominant_class:
        dominant_class_items = get_dominant_class_items(similar_meshes, dominant_class)
        top_10_enhanced = dominant_class_items.head(10)
    else:
        top_10_enhanced = top_10_regular

    # Calculate enhanced recall
    num_same_class_enhanced = top_10_enhanced[top_10_enhanced['obj_class'] == input_class].shape[0]
    recall_enhanced = num_same_class_enhanced / class_counts[input_class] if class_counts[input_class] > 0 else 0

    return input_class, recall_regular, recall_enhanced


def evaluate_retrieval_engine_with_class_normalization(descriptors_file, output_csv_regular, output_csv_enhanced):
    # Load the dataset descriptors
    dataset_df = pd.read_csv(descriptors_file)
    
    # Count the number of objects per class in the dataset
    class_counts = dataset_df['obj_class'].value_counts().to_dict()
    
    # Use multiprocessing to process each object in the dataset
    with Pool() as pool:
        results = pool.map(
            process_single_object,
            [(row, class_counts) for _, row in dataset_df.iterrows()]
        )
    
    # Initialize dictionaries to store per-class recall
    recall_per_class_regular = defaultdict(list)
    recall_per_class_enhanced = defaultdict(list)

    # Process results
    for input_class, recall_regular, recall_enhanced in results:
        recall_per_class_regular[input_class].append(recall_regular)
        recall_per_class_enhanced[input_class].append(recall_enhanced)

    # Calculate mean recall per class for regular and enhanced search
    mean_recall_per_class_regular = {cls: sum(recalls) / len(recalls) for cls, recalls in recall_per_class_regular.items()}
    mean_recall_per_class_enhanced = {cls: sum(recalls) / len(recalls) for cls, recalls in recall_per_class_enhanced.items()}

    # Save results to CSV
    regular_df = pd.DataFrame.from_dict(mean_recall_per_class_regular, orient='index', columns=['mean_recall_regular']).round(2)
    regular_df.index.name = 'obj_class'
    enhanced_df = pd.DataFrame.from_dict(mean_recall_per_class_enhanced, orient='index', columns=['mean_recall_enhanced']).round(2)
    enhanced_df.index.name = 'obj_class'
    
    enhanced_col = enhanced_df['mean_recall_enhanced']
    combined_recalls = pd.concat([regular_df, enhanced_col], axis=1)
    
    regular_df.to_csv(output_csv_regular)
    enhanced_df.to_csv(output_csv_enhanced)
    combined_recalls.to_csv('outputs/eval/combined_recalls.csv')
    
    print(f"Evaluation results saved with class-normalized recall: \n- Regular search: {output_csv_regular}\n- Enhanced search: {output_csv_enhanced}")


if __name__=="__main__":
    descriptors_file = "outputs/data/combined_descriptors.csv"
    output_csv_regular = "outputs/eval/evaluation_results_regular.csv"
    output_csv_enhanced = "outputs/eval/evaluation_results_enhanced.csv"
    evaluate_retrieval_engine_with_class_normalization(descriptors_file, output_csv_regular, output_csv_enhanced)
