import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict


def load_data(bigger_parent_folder: str, shuffled: bool = False):
    """
    Loads pairwise and summarised data from JSON files in given folder structure.
    """
    extracted_data = {}
    pairwise_id = 'pairwise' if not shuffled else 'pairwise-mismatched'

    for parent_folder in os.listdir(bigger_parent_folder):
        parent_folder_path = os.path.join(bigger_parent_folder, parent_folder)

        if os.path.isdir(parent_folder_path):
            extracted_data[parent_folder] = {'pairwise': [], 'summarised': []}

            # Load pairwise data
            pairwise_folder_path = os.path.join(parent_folder_path, pairwise_id)
            if os.path.isdir(pairwise_folder_path):
                for filename in os.listdir(pairwise_folder_path):
                    if filename.endswith('.json'):
                        with open(os.path.join(pairwise_folder_path, filename)) as f:
                            data = json.load(f)
                            df = pd.DataFrame(data)
                            similarities = [match['similarity'] for match in list(data.values())]
                            extracted_data[parent_folder]['pairwise'].append(
                                (filename.replace('.json', ''), len(df.columns), {'similarities': similarities}))

            # Load summarised data
            summarised_folder_path = os.path.join(parent_folder_path, 'summarised')
            if os.path.isdir(summarised_folder_path):
                for filename in os.listdir(summarised_folder_path):
                    if filename.endswith('.json'):
                        with open(os.path.join(summarised_folder_path, filename)) as f:
                            data = json.load(f)
                            df = pd.DataFrame(data)
                            extracted_data[parent_folder]['summarised'].append(
                                (filename.replace('.json', ''), len(df.columns)))

    return extracted_data


def calculate_hit_rates(summarised_data, pairwise_data):
    """
    Calculates Hit Rate |A∩B| / |A|.
    """
    model_dict = {model: value for model, value in summarised_data}
    model_pairs = [(model_pair, pairwise_count / model_dict.get(model_pair.split('-')[-1], 1))
                   for model_pair, pairwise_count, _ in pairwise_data]
    return np.array(model_pairs, dtype=[('model_pair', 'U50'), ('hit_rate', 'f4')])


def calculate_sso_coefficients(summarised_data, pairwise_data):
    """
    Calculates Szymkiewicz–Simpson Overlap Coefficient |A∩B| / min(|A|, |B|).
    """
    model_dict = {model: value for model, value in summarised_data}
    sso_coefficients = [
        (model_pair, pairwise_count / min(model_dict.get(model_name, 1), model_dict.get(reviewer_name, 1)))
        for model_pair, pairwise_count, _ in pairwise_data
        if (model_name := model_pair.split('-')[-1]) and (reviewer_name := model_pair.split('-')[0])
    ]
    return np.array(sso_coefficients, dtype=[('model_pair', 'U50'), ('sso_coefficient', 'f4')])


def calculate_jaccard_indices(summarised_data, pairwise_data):
    """
    Calculates Jaccard Index |A∩B| / |A∪B|.
    """
    model_dict = {model: value for model, value in summarised_data}
    jaccard_indices = [
        (model_pair,
         pairwise_count / (model_dict.get(model_name, 0) + model_dict.get(reviewer_name, 0) - pairwise_count))
        for model_pair, pairwise_count, _ in pairwise_data
        if (model_name := model_pair.split('-')[-1]) and (reviewer_name := model_pair.split('-')[0])
    ]
    return np.array(jaccard_indices, dtype=[('model_pair', 'U50'), ('jaccard_index', 'f4')])


def calculate_sd_coefficient(summarised_data, pairwise_data):
    """
    Calculates Sørensen–Dice Coefficient 2|A∩B| / (|A| + |B|).
    """
    model_dict = {model: value for model, value in summarised_data}
    sd_coefficients = [
        (model_pair, 2 * pairwise_count / (model_dict.get(model_name, 0) + model_dict.get(reviewer_name, 0)))
        for model_pair, pairwise_count, _ in pairwise_data
        if (model_name := model_pair.split('-')[-1]) and (reviewer_name := model_pair.split('-')[0])
    ]
    return np.array(sd_coefficients, dtype=[('model_pair', 'U50'), ('sd_coefficient', 'f4')])


def calculate_recall(summarised_data, pairwise_data, similarity_threshold: int = 7):
    """
    Calculates recall as the ratio of relevant matches (above similarity threshold)
    to the total points in the human review data.
    """
    # Create a dictionary from summarised_data where the model name is the key and the value is its total count
    human_comments = {model: value for model, value in summarised_data if 'review' in model}

    # Get amount of similarities where above similarity threshold
    similar_overlaps = {
        model_pair: sum(1 for value in similarities['similarities'] if int(value) >= similarity_threshold)
        for model_pair, pairwise_count, similarities in pairwise_data
    }

    # Get similarity only for human
    similar_overlaps_human = {pair: (count if 'review' in pair else 0) for pair, count in similar_overlaps.items()}

    # The total number of points in human review data is equivalent to the total number of models (or reviewers)
    human_comments_total = sum(human_comments.values())

    # Relevant matches over total points in the human review data
    recall_data = [
        (pair, count / human_comments_total if human_comments_total > 0 else 0)
        for pair, count in similar_overlaps_human.items()
    ]

    return np.array(recall_data, dtype=[('model_pair', 'U50'), ('recall', 'f4')])


def combine_metrics(hit_rates, sso_coefficients, jaccard_indices, sd_coefficients):
    """
    Combines calculated metrics into a single dictionary.
    """
    all_metrics = {}
    for model_pair, hit_rate in hit_rates:
        all_metrics[model_pair] = {'hit_rate': hit_rate}

    for model_pair, sso in sso_coefficients:
        all_metrics[model_pair]['sso_coefficient'] = sso

    for model_pair, jaccard in jaccard_indices:
        all_metrics[model_pair]['jaccard_index'] = jaccard

    for model_pair, sd in sd_coefficients:
        all_metrics[model_pair]['sd_coefficient'] = sd

    return all_metrics


def categorize_comparison(comparison):
    """Categorize comparisons based on names."""

    comparison_first = comparison.split('-')[0]
    comparison_last = comparison.split('-')[-1]

    # Get GPT Scores
    if "gpt4" in comparison and "review" in comparison:
        return "gpt4-vs-human"
    # Get Gemini Scores
    elif "gemini_pro" in comparison and "review" in comparison:
        return "gemini-vs-human"
    # Get Claude Scores
    elif "claude_opus" in comparison and "review" in comparison:
        return "claude-vs-human"
    # Get Human Scores
    elif "review" in comparison_first and "review" in comparison_last:
        return "human-vs-human"
    # Get LLM Scores
    elif (comparison_first in ["gpt4", "gemini_pro", "claude_opus"]) and (
            comparison_last in ["gpt4", "gemini_pro", "claude_opus"]):
        return "llm-vs-llm"
    return None


def check_llm_vs_human_category(comparison):
    # Define the pairs to check for each model
    models = ["gpt4", "gemini_pro", "claude_opus"]

    # Check if "review" exists and at least one model is mentioned
    if "review" in comparison and any(model in comparison for model in models):
        return "llm-vs-human"
    return None


def check_averaged_category(comparison):
    # Check if category is one of the averaged groups.
    if "average" in comparison:
        return comparison
    return None


def average_across_categories(all_metrics):
    """
    Average Metrics across all papers and all comparisons for:
    - GPT vs. Human Reviews
    - Gemini vs. Human Reviews
    - Claude vs. Human Reviews
    - Humans vs. Humans
    - LLMs vs. LLMs

    :param all_metrics: Dictionary containing metrics for each paper and comparison.
    :return: Updated all_metrics with averages for each comparison category.
    """

    # Initialize separate dictionary to accumulate metrics across all comparisons
    accumulative_metrics_by_category = defaultdict(lambda: defaultdict(list))

    # Iterate over all papers and comparisons to accumulate metrics
    for paper, comparisons in all_metrics.items():
        for comparison, metrics in comparisons.items():
            # Determine the category of the comparison
            category = categorize_comparison(comparison)
            llm_vs_human = check_llm_vs_human_category(comparison)

            if category is not None:
                # Accumulate values for each metric in the category
                for metric, value in metrics.items():
                    accumulative_metrics_by_category[category][metric].append(value)
            if llm_vs_human is not None:
                # Accumulate values for each metric in the category
                for metric, value in metrics.items():
                    accumulative_metrics_by_category[llm_vs_human][metric].append(value)

    # Calculate averages and store them in a separate dictionary
    average_metrics_by_category = {}
    for category, metrics in accumulative_metrics_by_category.items():
        # Compute the average for each metric in the category
        average_metrics_by_category[category] = {metric: sum(values) / len(values) for metric, values in
                                                 metrics.items()}

    # Append average metrics to `all_metrics` under each paper with unique category keys
    for paper in all_metrics.keys():
        for category, average_metrics in average_metrics_by_category.items():
            comparison_name = f"{category}-average"
            all_metrics[paper][comparison_name] = average_metrics

    return all_metrics


def sort_and_swap(dictionary, pos1=-2, pos2=-3):
    """
    Sorts the dictionary and swaps two specified items.

    Parameters:
        dictionary (dict): The dictionary to sort and swap.
        pos1 (int): The position of the first item to swap (default is -2).
        pos2 (int): The position of the second item to swap (default is -3).

    Returns:
        dict: A new dictionary with sorted items and specified items swapped.
    """
    # Sort the dictionary
    sorted_dict = dict(sorted(dictionary.items()))

    # Convert to a list to manipulate positions
    sorted_list = list(sorted_dict.items())

    # Swap the specified positions
    sorted_list[pos1], sorted_list[pos2] = sorted_list[pos2], sorted_list[pos1]

    # Convert back to dictionary and return
    return dict(sorted_list)


def average_across_papers(all_comparisons):
    """
    Calculate the average metrics across all papers for each comparison category.

    :param all_comparisons: Dictionary of pages, each containing comparisons with metrics and their values.
    :return: Dictionary with average values of each metric, grouped by comparison category.
    """

    # Initialize a dictionary to store sums and counts for each category
    category_totals = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'count': 0}))

    # Accumulate sums and counts for each metric by category
    for paper, comparisons in all_comparisons.items():
        for comparison, metrics in comparisons.items():

            # Determine the category of the comparison
            category = check_averaged_category(comparison)
            if category is None:
                continue

            # Accumulate metrics for the identified category
            for metric, value in metrics.items():
                category_totals[category][metric]['sum'] += value
                category_totals[category][metric]['count'] += 1

    # Sort Categories for nicer plots
    category_totals = sort_and_swap(category_totals)

    # Calculate averages for each metric in each category
    average_metrics_by_category = {
        category: {metric: totals['sum'] / totals['count'] for metric, totals in metrics.items()}
        for category, metrics in category_totals.items()
    }

    return average_metrics_by_category


def save_metrics(data, file_path):
    """Save metrics data to a JSON file, converting any non-JSON serializable types."""

    # Convert any numpy data types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    # Convert the data
    data = convert_types(data)

    # Write to JSON file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to '{file_path}'")


def main():
    bigger_parent_folder = "./extracted"
    extracted_data = load_data(bigger_parent_folder, shuffled=False)

    all_metrics = {}
    for paper, data in extracted_data.items():
        if data['summarised'] and data['pairwise']:
            hit_rates = calculate_hit_rates(data['summarised'], data['pairwise'])
            sso_coefficients = calculate_sso_coefficients(data['summarised'], data['pairwise'])
            jaccard_indices = calculate_jaccard_indices(data['summarised'], data['pairwise'])
            sd_coefficients = calculate_sd_coefficient(data['summarised'], data['pairwise'])

            all_metrics[paper] = combine_metrics(hit_rates, sso_coefficients, jaccard_indices, sd_coefficients)

    print(all_metrics)

    # Compute average across Categories ('gpt4-vs-human', 'llm-vs-llm' ...)
    metrics_data = average_across_categories(all_metrics)
    average_data = average_across_papers(metrics_data)

    # Write Metrics for Plotting
    save_metrics(metrics_data, f'processed/all_metrics.json')
    save_metrics(average_data, f'processed/average_metrics.json')

    print(f"Comparisons evaluated and saved under '/processed'.")


# Run the main function
if __name__ == "__main__":
    main()
