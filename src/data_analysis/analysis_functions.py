import os

import pandas as pd
import scipy.stats as stats
import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox, yeojohnson, shapiro, levene, mannwhitneyu

from src.utils.config import BASE_DIR


def calculate_frequency(force_data, sampling_rate):
    sample_num = len(force_data)

    # Check if there are enough samples to perform FFT
    if sample_num < 2:
        return 0

    force_fft = fft(force_data)
    freqs = fftfreq(sample_num, 1 / sampling_rate)
    fft_magnitude = np.abs(force_fft)

    positive_freqs = freqs[:sample_num // 2]
    positive_fft_magnitude = fft_magnitude[:sample_num // 2]

    dominant_freq_index = np.argmax(positive_fft_magnitude[1:])

    return positive_freqs[dominant_freq_index + 1]


def calculate_metrics(participants_data, metrics_list=None, sampling_rate=30):
    participant_metrics = {}

    for participant, values in participants_data.items():

        if values:
            values = np.array(values)
            participant_metrics[participant] = {}

            if 'mean' in metrics_list:
                participant_metrics[participant]['mean'] = np.mean(values)
            if 'std' in metrics_list:
                participant_metrics[participant]['std'] = np.std(values)
            if 'frequency' in metrics_list:
                # Calculate dominant frequency using the FFT function
                participant_metrics[participant]['frequency'] = calculate_frequency(values, sampling_rate)
            if 'rate_of_change' in metrics_list:
                # Calculate rate of change of the control position (how quickly the control is being moved)
                delta_t = 1 / sampling_rate
                participant_metrics[participant]['rate_of_change'] = np.mean(np.abs(np.diff(values)) / delta_t)
            if 'control_reversals' in metrics_list:
                # Count the number of times the control reverses direction (i.e., sign change in the derivative)
                participant_metrics[participant]['control_reversals'] \
                    = np.sum(np.diff(np.sign(np.diff(values))) != 0)
            if 'time_in_neutral' in metrics_list:
                # Calculate the proportion of time the control is within a neutral range around zero
                participant_metrics[participant]['time_in_neutral'] = np.sum(np.abs(values) <= 0.05) / len(
                    values)
            if 'control_smoothness' in metrics_list:
                # Calculate control smoothness by measuring the jerk (third derivative of position)
                delta_t = 1 / sampling_rate
                control_velocity = np.diff(values) / delta_t
                control_acceleration = np.diff(control_velocity) / delta_t
                control_jerk = np.diff(control_acceleration) / delta_t
                participant_metrics[participant]['control_smoothness'] = np.mean(np.abs(control_jerk))
            if 'slip_duration' in metrics_list:
                neutral_range = (-0.05, 0.05)  # Example for slip indicator
                participant_metrics[participant]['slip_duration'] = np.sum((values < neutral_range[0]) | (values > neutral_range[1])) / len(values)
            if 'coefficient_of_variation' in metrics_list:
                participant_metrics[participant]['coefficient_of_variation'] = np.std(values) / np.mean(values)

    return participant_metrics


def calculate_distribution(participants_data, column, novice_pilots, experienced_pilots):
    novice_pilots_data = []
    experienced_pilots_data = []
    min_trial_info = {'force': float('inf'), 'trial': None}
    max_trial_info = {'force': float('-inf'), 'trial': None}

    for participant, trials in participants_data.items():
        if participant in novice_pilots:
            curr = novice_pilots_data
        elif participant in experienced_pilots:
            curr = experienced_pilots_data
        else:
            continue
        for trial, df in trials.items():
            if column in df:
                forces = df[column].values
                curr.extend(forces)

                trial_min_force = np.min(forces)
                trial_max_force = np.max(forces)

                if trial_min_force < min_trial_info['force']:
                    min_trial_info['force'] = trial_min_force
                    min_trial_info['trial'] = (participant, trial)

                if trial_max_force > max_trial_info['force']:
                    max_trial_info['force'] = trial_max_force
                    max_trial_info['trial'] = (participant, trial)

    novice_pilots_data = np.array(novice_pilots_data)
    experienced_pilots_data = np.array(experienced_pilots_data)

    novice_stats = {
        'mean': np.mean(novice_pilots_data),
        'median': np.median(novice_pilots_data),
        'std': np.std(novice_pilots_data),
        'skewness': stats.skew(novice_pilots_data)
    }

    experienced_stats = {
        'mean': np.mean(experienced_pilots_data),
        'median': np.median(experienced_pilots_data),
        'std': np.std(experienced_pilots_data),
        'skewness': stats.skew(experienced_pilots_data)
    }

    print(f"{column} - Data Statistics (Novice Pilots):")
    print(
        f"Mean: {novice_stats['mean']}\nMedian: {novice_stats['median']}\nStd: {novice_stats['std']}\nSkewness: {novice_stats['skewness']}")
    print("\n")
    print(f"{column} - Data Statistics (Experienced Pilots):")
    print(
        f"Mean: {experienced_stats['mean']}\nMedian: {experienced_stats['median']}\nStd: {experienced_stats['std']}\nSkewness: {experienced_stats['skewness']}")
    print("\n")
    print(f"Global minimum force: {min_trial_info['force']} occurred in trial {min_trial_info['trial']}")
    print(f"Global maximum force: {max_trial_info['force']} occurred in trial {max_trial_info['trial']}")

    plt.figure(figsize=(16, 6))

    # Novice pilots plot
    plt.subplot(1, 2, 1)
    sns.kdeplot(novice_pilots_data, fill=True, color='green')
    plt.title(f'Novice Pilots - {column} Distribution')
    plt.xlabel('Force')
    plt.ylabel('Density')

    # Experienced pilots plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(experienced_pilots_data, fill=True, color='blue')
    plt.title(f'Experienced Pilots - {column} Distribution')
    plt.xlabel('Force')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "parameters", "distribution", "raw", f'{column}_distribution.png'))
    plt.show()

    return novice_stats, experienced_stats, min_trial_info, max_trial_info


def visualize_distribution_after_normalization(novice_data, experienced_data, column):
    novice_pilots_data = []
    experienced_pilots_data = []
    min_trial_info = {'force': float('inf'), 'trial': None}
    max_trial_info = {'force': float('-inf'), 'trial': None}

    for participant, df in novice_data.items():
        novice_pilots_data.extend(df)

    for participant, df in experienced_data.items():
        experienced_pilots_data.extend(df)

    novice_pilots_data = np.array(novice_pilots_data)
    experienced_pilots_data = np.array(experienced_pilots_data)

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    sns.kdeplot(novice_pilots_data, fill=True, color='green')
    plt.title(f'Novice Pilots - {column} Distribution')
    plt.xlabel('Force')
    plt.ylabel('Density')

    plt.subplot(1, 2, 2)
    sns.kdeplot(experienced_pilots_data, fill=True, color='blue')
    plt.title(f'Experienced Pilots - {column} Distribution')
    plt.xlabel('Force')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "parameters", "distribution", "normalized", f'{column}_distribution_normalized.png'))
    plt.show()


def perform_t_test(novice_values, experienced_values):
    """Perform an independent t-test between novice and experienced values."""
    t_stat, p_value = stats.ttest_ind(novice_values, experienced_values, equal_var=False)
    return t_stat, p_value


def normalize_data(data, normalization_type):
    data = np.array(data)

    if normalization_type == 'z-score':
        return (data - np.mean(data)) / np.std(data)

    elif normalization_type == 'min-max':
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    elif normalization_type == 'log-z-score':
        # Shift data if there are negative or zero values
        if np.any(data <= 0):
            data += np.abs(np.min(data)) + 1
        data = np.log(data)
        return (data - np.mean(data)) / np.std(data)

    elif normalization_type == 'square-root':
        # Shift data if there are negative values
        if np.any(data < 0):
            data += np.abs(np.min(data))
        data = np.sqrt(data)
        return (data - np.mean(data)) / np.std(data)

    elif normalization_type == 'box-cox':
        # Shift data if there are negative values
        if np.any(data <= 0):
            data += np.abs(np.min(data)) + 1
        data, _ = boxcox(data)  # Perform Box-Cox transformation
        return (data - np.mean(data)) / np.std(data)
    elif normalization_type == 'yeo-johnson':
        data, _ = yeojohnson(data)  # Perform Yeo-Johnson transformation
        return (data - np.mean(data)) / np.std(data)
    elif normalization_type == 'sine-cosine':
        # Apply sine and cosine transformation for cyclic data
        sine_data = np.sin(data)
        cosine_data = np.cos(data)
        return sine_data, cosine_data
    elif normalization_type == 'no':
        return data
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")


def check_t_test_applicability(novice_data, experienced_data):
    # Check for normality using Shapiro-Wilk test
    _, p_novice = shapiro(novice_data)
    _, p_experienced = shapiro(experienced_data)

    # Check for equal variances using Levene's test
    _, p_levene = levene(novice_data, experienced_data)

    # Conditions for t-test:
    t_test_possible = p_novice > 0.05 and p_experienced > 0.05 and p_levene > 0.05

    return t_test_possible, p_novice, p_experienced, p_levene


def structure_participant_data(participants_data, novice_pilots, experienced_pilots, column_name):
    novice_data = {}
    experienced_data = {}

    for pilot, trials in participants_data.items():
        values = []
        for trial, df in trials.items():
            if column_name in df:
                if pilot in novice_pilots:
                    values.extend(df[column_name].values)
                elif pilot in experienced_pilots:
                    values.extend(df[column_name].values)
        if pilot in novice_pilots:
            novice_data[pilot] = values
        elif pilot in experienced_pilots:
            experienced_data[pilot] = values

    return novice_data, experienced_data


def structure_and_normalize_participant_data(participants_data, novice_pilots, experienced_pilots, column_name, normalization_type):
    novice_data = []
    experienced_data = []

    print(f"{column_name}")

    for pilot, trials in participants_data.items():
        for trial, df in trials.items():
            if column_name in df:
                if pilot in novice_pilots:
                    novice_data.extend(df[column_name].values)
                elif pilot in experienced_pilots:
                    experienced_data.extend(df[column_name].values)

    normalized_novice_data = normalize_data(novice_data, normalization_type)
    normalized_experienced_data = normalize_data(experienced_data, normalization_type)

    normalized_data_dict = {'novice': {}, 'experienced': {}}
    group_data = {'novice': (novice_pilots, normalized_novice_data),
                  'experienced': (experienced_pilots, normalized_experienced_data)}

    for group, (pilots, normalized_data) in group_data.items():
        index = 0
        for pilot in pilots:
            if pilot in participants_data:
                pilot_normalized_values = []
                for trial, df in participants_data[pilot].items():
                    trial_length = len(df[column_name])
                    pilot_normalized_values.extend(normalized_data[index: index + trial_length])
                    index += trial_length
                normalized_data_dict[group][pilot] = pilot_normalized_values

    novice_data_normalized = normalized_data_dict['novice']
    experienced_data_normalized = normalized_data_dict['experienced']

    t_test_possible, p_novice, p_experienced, p_levene = check_t_test_applicability(normalized_novice_data, normalized_experienced_data)

    print(f"Shapiro-Wilk Test for Novice (p-value): {p_novice}")
    print(f"Shapiro-Wilk Test for Experienced (p-value): {p_experienced}")
    print(f"Levene's Test for Equal Variances (p-value): {p_levene}")
    print(f"Is t-test possible? {'Yes' if t_test_possible else 'No'}")

    return novice_data_normalized, experienced_data_normalized


def perform_mann_whitney_u_test(novice_values, experienced_values):
    u_stat, p_value = mannwhitneyu(novice_values, experienced_values, alternative='two-sided')
    direction = "Expert > Novice" if np.median(experienced_values) > np.median(novice_values) else "Expert < Novice"
    return u_stat, p_value, direction


def find_median_trial(participants_data, novice_pilots, experienced_pilots, column_name):
    closest_trials = {'novice': {}, 'experienced': {}}

    novice_data, experienced_data = structure_participant_data(participants_data, novice_pilots, experienced_pilots,
                                                               column_name)

    def find_closest_trial(pilot_data, pilot_trials, column_name):
        # Calculate the overall median for the pilot's data across trials
        overall_median = np.median(pilot_data)

        # Track the closest trial to the median
        closest_trial = None
        closest_distance = float('inf')

        # Iterate over trials to find the trial closest to the median
        for trial_id, df in pilot_trials.items():
            if column_name in df.columns:
                trial_median = np.median(df[column_name].values)
                distance = abs(trial_median - overall_median)

                if distance < closest_distance:
                    closest_distance = distance
                    closest_trial = trial_id

        return closest_trial

    # Process each novice pilot
    for pilot, trials in participants_data.items():
        if pilot in novice_pilots:
            closest_trials['novice'][pilot] = find_closest_trial(novice_data[pilot], trials, column_name)

    # Process each experienced pilot
    for pilot, trials in participants_data.items():
        if pilot in experienced_pilots:
            closest_trials['experienced'][pilot] = find_closest_trial(experienced_data[pilot], trials, column_name)

    return closest_trials


def find_median_trial_across_group(participants_data, novice_pilots, experienced_pilots, column_name):
    novice_data, experienced_data = structure_participant_data(participants_data, novice_pilots, experienced_pilots,
                                                               column_name)

    novice_all_values = np.concatenate(list(novice_data.values()))
    experienced_all_values = np.concatenate(list(experienced_data.values()))

    novice_median = np.median(novice_all_values)
    experienced_median = np.median(experienced_all_values)

    # Dictionary to store the closest trial for each group
    closest_trials = {'novice': None, 'experienced': None}
    min_distances = {'novice': float('inf'), 'experienced': float('inf')}

    # Helper function to find the closest trial to the group median for a given pilot
    def find_closest_trial(pilot_trials, group_median):
        closest_trial_id = None
        closest_distance = float('inf')

        for trial_name, df in pilot_trials.items():
            if column_name in df.columns:
                trial_median = np.median(df[column_name].values)
                distance = abs(trial_median - group_median)

                if distance < closest_distance:
                    closest_distance = distance
                    closest_trial_id = trial_name
        return closest_trial_id, closest_distance

    # Find the closest trial to the median for the novice group
    for pilot, trials in participants_data.items():
        if pilot in novice_pilots:
            trial_id, distance = find_closest_trial(trials, novice_median)
            if distance < min_distances['novice']:
                min_distances['novice'] = distance
                closest_trials['novice'] = (pilot, trial_id)

    # Find the closest trial to the median for the experienced group
    for pilot, trials in participants_data.items():
        if pilot in experienced_pilots:
            trial_id, distance = find_closest_trial(trials, experienced_median)
            if distance < min_distances['experienced']:
                min_distances['experienced'] = distance
                closest_trials['experienced'] = (pilot, trial_id)

    return closest_trials


