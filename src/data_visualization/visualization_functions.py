import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


from src.data_analysis.analysis_functions import perform_mann_whitney_u_test, perform_t_test
from src.utils.config import BASE_DIR

all_results_df = pd.DataFrame()
significant_results_df = pd.DataFrame()


def visualize_mann_whitney_u_tests(novice_metrics, experienced_metrics, metrics_to_compare, rows, cols, column_name):
    print(f"Running Mann-Whitney U tests for {column_name} metrics...")
    global all_results_df
    global significant_results_df
    test_results = []
    significant_results = []
    data = []
    for metric_name, unit in metrics_to_compare.items():
        for group, metrics in [('Novice', novice_metrics), ('Experienced', experienced_metrics)]:
            for participant, values in metrics.items():
                data.append({
                    'Participant': participant,
                    'Pilot Experience Level': group,
                    'Metric': metric_name,
                    unit: values[metric_name]
                })

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))
    axes = axes.flatten()

    for i, (metric_name, unit) in enumerate(metrics_to_compare.items()):
        subset = df[df['Metric'] == metric_name]

        sns.violinplot(x='Pilot Experience Level', y=unit, data=subset, ax=axes[i], hue='Pilot Experience Level',
                       palette='Set2', legend=False)

        novice_values = [metrics[metric_name] for metrics in novice_metrics.values()]
        experienced_values = [metrics[metric_name] for metrics in experienced_metrics.values()]

        if novice_values and experienced_values:
            u_stat, u_p_value, direction = perform_mann_whitney_u_test(novice_values, experienced_values)
            t_stat, t_p_value = perform_t_test(novice_values, experienced_values)

            print(
                f"Mann-Whitney U test for {metric_name} ({column_name}): U-statistic = {u_stat}, p-value = {u_p_value}")
            print(f"T-test for {metric_name} ({column_name}): t-statistic = {t_stat}, p-value = {t_p_value}")

            axes[i].set_title(f"{metric_name.capitalize()} ({column_name})")

            # for u test results and p test results
            # f"U-stat: {u_stat:.2f}\nU-P-value: {u_p_value:.6f}\nT-stat: {t_stat:.2f}\nT-P-value: {t_p_value:.6f}"

            axes[i].annotate(f"U-stat: {u_stat:.2f}\nU-P-value: {u_p_value:.6f}",
                             xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='top',
                             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

            test_results.append({
                'Column': column_name,
                'Metric': metric_name,
                'Mann-Whitney U-stat': u_stat,
                'Mann-Whitney P-value': u_p_value,
                'Direction': direction,
                'T-stat': t_stat,
                'T-test P-value': t_p_value
            })

            if u_p_value <= 0.05:
                significant_results.append({
                    'Column': column_name,
                    'Metric': metric_name,
                    'Direction': direction,
                    'Mann-Whitney U-stat': u_stat,
                    'Mann-Whitney P-value': u_p_value,
                    'T-stat': t_stat,
                    'T-test P-value': t_p_value
                })
                curr_fig, curr_ax = plt.subplots(figsize=(6, 6))
                sns.violinplot(x='Pilot Experience Level', y=unit, data=subset, ax=curr_ax, hue='Pilot Experience Level',
                               palette='Set2', legend=False)
                curr_ax.set_title(f"Significant Result: {metric_name.capitalize()} ({column_name})")
                curr_ax.annotate(
                    f"U-stat: {u_stat:.2f}\nU-P-value: {u_p_value:.6f}",
                    xy=(0.5, 0.95), xycoords='axes fraction', ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

                significant_folder = os.path.join(BASE_DIR, "results", "significant_results")
                os.makedirs(significant_folder, exist_ok=True)

                plot_filename = f"{column_name}_significant_{metric_name}_statistics.png"
                curr_fig.savefig(os.path.join(significant_folder, plot_filename))
                plt.close(curr_fig)

        else:
            print(f"Not enough data to perform Mann-Whitney U test for {metric_name} ({column_name}).")
            axes[i].set_title(f"{metric_name.capitalize()}: Not enough data")

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "results", f'{column_name}_statistics.png'))

    plt.show()

    all_results_df = pd.concat([all_results_df, pd.DataFrame(test_results)], ignore_index=True)
    significant_results_df = pd.concat([significant_results_df, pd.DataFrame(significant_results)], ignore_index=True)


def save_all_results():
    global all_results_df

    # Save the accumulated test results from all columns to a single CSV file
    if not all_results_df.empty:
        all_results_df.to_csv(os.path.join(BASE_DIR, "results", "all_results.csv"), index=False)
    if not significant_results_df.empty:
        significant_results_df.to_csv(
            os.path.join(BASE_DIR, "results", "significant_results", "significant_results.csv"),
            index=False)
    else:
        print("No test results to save.")


def plot_time_series_for_median_trials(participants_data, closest_trials, column_name, parameter_name, y_unit, sampling_rate=30):
    """
    Plots the time series for the closest-to-median trials in novice and experienced groups.

    Parameters:
        participants_data (dict): Nested dictionary of data {pilot_id: {trial_id: DataFrame}}.
        closest_trials (dict): Dictionary with the trial closest to the median for novice and experienced groups.
        column_name (str): Name of the parameter column to plot as a time series.
        sampling_rate (int): The sampling rate of the data in Hz (default is 30 Hz).
    """
    # Calculate time step based on sampling rate
    time_step = 1 / sampling_rate

    plt.figure(figsize=(3.1, 3.6))

    # Plot for Novice Group
    novice_pilot, novice_trial = closest_trials['novice']
    novice_df = participants_data[novice_pilot][novice_trial]

    if column_name in novice_df.columns:
        time_values_novice = np.arange(len(novice_df[column_name])) * time_step
        plt.plot(time_values_novice, novice_df[column_name],
                 label=f"Novice", color='#66c2a5')

    # Plot for Experienced Group
    experienced_pilot, experienced_trial = closest_trials['experienced']
    experienced_df = participants_data[experienced_pilot][experienced_trial]

    if column_name in experienced_df.columns:
        time_values_experienced = np.arange(len(experienced_df[column_name])) * time_step
        plt.plot(time_values_experienced, experienced_df[column_name],
                 label=f"Experienced", color='#fc8d62')

    # plt.xlabel(fontsize=2)
    # plt.ylabel(fontsize=0.5)
    # plt.title(f"Time Series of {parameter_name} for Novice and Experienced Pilots", fontsize=2)
    plt.xticks(fontsize=7)                    # Set size for X-axis tick labels
    plt.yticks(fontsize=7)                    # Set size for Y-axis tick labels
    plt.legend(fontsize=7, loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
