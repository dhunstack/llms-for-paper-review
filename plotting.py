import json
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('macosx')
SEABORN_THEME = "ticks"
COLOR_PALETTE = 'Pastel2'

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style=SEABORN_THEME, rc=custom_params)

TITLE_FONTSIZE = 14
Y_LABEL_FONTSIZE = 14
X_LABEL_FONTSIZE = 12
Y_LIM = 35


def load_metrics(file_path):
    """Load metrics data from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def plot_avg_of_dataset(average_data, metrics: list = None):
    """
    Plot the average metrics for each category with bars for each metric, scaled to percentages,
    with clean labels in title case.

    :param average_data: Dictionary of average metrics by category.
    :param metrics: Optional list of metrics to plot.
    """

    # Check if average_data is empty
    if not average_data:
        print("No data available to plot.")
        return

    # Extract categories and metrics
    categories = list(average_data.keys())
    all_metrics = list(next(iter(average_data.values())).keys())

    # Metrics for plotting
    metrics, metric_names = (metrics, "_".join(metrics)) if metrics else (all_metrics, 'all_metrics')

    # Data for plotting, scaled to %
    values = {metric: [average_data[category][metric] * 100 for category in categories] for metric in metrics}

    # Convert metric names to title case with spaces
    metric_labels = [metric.replace('_', ' ').title() for metric in metrics]

    # Set up the figure with moderate aspect ratio
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define a custom pastel color palette with Seaborn
    colors = sns.color_palette(COLOR_PALETTE, len(metrics))

    # Set bar width and x positions with extra spacing between groups
    width = 0.15
    x = np.arange(len(categories)) * (len(metrics) * width + 0.2)

    # Plot bars for each metric with formatted metric labels
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax.bar(x + i * width, values[metric], width=width, label=label, color=colors[i])

    # Formatting
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=X_LABEL_FONTSIZE)
    ax.set_ylabel('Metric Value (%)', fontsize=Y_LABEL_FONTSIZE)
    ax.set_title('Metrics for Dataset', fontsize=TITLE_FONTSIZE)

    # Set y-axis limit to 0-100%
    # ax.set_ylim(0, Y_LIM, 0, 100)
    ax.set_ylim(0, Y_LIM)

    # Remove background grid lines but keep borders
    sns.despine(left=False, bottom=False)

    # Customize the legend and layout
    ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(pad=3.0)

    # Save the plot as a PNG file
    plt.savefig(f'figures/avg_dataset_{metric_names}.png', format='png', dpi=300)

    plt.show()


def plot_avg_of_all_metrics(data):
    """
    Plot the average of all metrics for each category as a single bar, scaled to percentages,
    using colors from the Pastel2 colormap in Seaborn for a clean aesthetic.

    :param data: Dictionary of average metrics by category.
    """
    # Compute the average metric for each category, scaled to %
    categories = list(data.keys())
    avg_values = [np.mean(list(metrics.values())) * 100 for metrics in data.values()]

    # Create a DataFrame for Seaborn compatibility
    df = pd.DataFrame({'Category': categories, 'Average (%)': avg_values})

    # Set up the figure and axis with a taller aspect ratio
    plt.figure(figsize=(5, 7))

    # Create a Seaborn bar plot with pastel color palette
    sns.barplot(
        data=df,
        x='Category',
        y='Average (%)',
        palette=COLOR_PALETTE
    )

    # Remove background grid lines but keep the border lines
    sns.despine(left=False, bottom=False)

    # Formatting
    plt.ylim(0, Y_LIM, 0, 100)
    plt.title('Average Metric Percentage by Category', fontsize=TITLE_FONTSIZE)
    plt.xlabel('')
    plt.ylabel('Metric Average (%)', fontsize=Y_LABEL_FONTSIZE)
    plt.xticks(rotation=45, ha='right', fontsize=X_LABEL_FONTSIZE)

    # Save the plot
    plt.tight_layout()
    plt.savefig('figures/avg_metrics.png', format='png', dpi=300)
    plt.show()


if __name__ == "__main__":

    # Define file paths
    metrics_file_path = 'processed/all_metrics.json'
    average_file_path = 'processed/average_metrics.json'

    # Load data
    metrics_data = load_metrics(metrics_file_path)
    average_data = load_metrics(average_file_path)

    # Plot data
    plot_avg_of_dataset(average_data)
    plot_avg_of_dataset(average_data, metrics=['hit_rate'])
    plot_avg_of_dataset(average_data, metrics=['sso_coefficient'])
    plot_avg_of_dataset(average_data, metrics=['jaccard_index'])
    plot_avg_of_dataset(average_data, metrics=['sd_coefficient'])
    plot_avg_of_all_metrics(average_data)

    print(f'Plots created and saved under /figures.')
