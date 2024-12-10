import numpy as np
import pandas as pd

from src.utils.config import COLUMNS_DIR
from src.utils.data_loader import get_headers
from scipy.stats import spearmanr


def prepare_data_for_correlation(participants_data, novice_pilots, experienced_pilots):
    targeted_parameters = get_headers(COLUMNS_DIR)
    novice_data = {param: [] for param in targeted_parameters}
    experienced_data = {param: [] for param in targeted_parameters}

    for pilot, trials in participants_data.items():
        for trial, df in trials.items():
            if pilot in novice_pilots:
                target_data = novice_data
            elif pilot in experienced_pilots:
                target_data = experienced_data

            for param in targeted_parameters:
                if param in df.columns:
                    target_data[param].extend(df[param].values)

    for param in targeted_parameters:
        novice_data[param] = list(novice_data[param])
        experienced_data[param] = list(experienced_data[param])

    return {'novice': novice_data, 'experienced': experienced_data}


def calculate_two_parameter_correlations(prepared_data, method='spearman'):
    correlations = {}

    for group in ['novice', 'experienced']:
        data = prepared_data[group]
        df = pd.DataFrame(data)

        if method == 'spearman':
            correlation_matrix = df.corr(method='spearman')
        else:
            correlation_matrix = df.corr(method='kendall')

        correlations[group] = correlation_matrix

    return correlations


def print_significant_correlations(correlations, threshold=0.4):
    for group in correlations:
        print(f"\n{'=' * 40}\n Correlation Matrix for {group.capitalize()} Group\n{'=' * 40}")
        corr_matrix = correlations[group]

        printed_pairs = set()

        for param1 in corr_matrix.columns:
            for param2 in corr_matrix.index:
                if param1 != param2 and (param2, param1) not in printed_pairs:
                    correlation_value = corr_matrix.loc[param1, param2]
                    if abs(correlation_value) >= threshold:
                        print(f"Correlation between {param1} and {param2}: {correlation_value:.3f}")
                        printed_pairs.add((param1, param2))


def calculate_partial_correlation(x, y, z):
    x_resid = x - np.polyfit(z, x, 1)[0] * z
    y_resid = y - np.polyfit(z, y, 1)[0] * z

    partial_corr, _ = spearmanr(x_resid, y_resid)
    return partial_corr


def calculate_three_parameter_correlations(group_data, parameters):

    x = np.array(group_data[parameters[0]])
    y = np.array(group_data[parameters[1]])
    z = np.array(group_data[parameters[2]])

    results = {
        f"{parameters[0]} and {parameters[1]} controlling for {parameters[2]}": calculate_partial_correlation(x, y, z),
        f"{parameters[0]} and {parameters[2]} controlling for {parameters[1]}": calculate_partial_correlation(x, z, y),
        f"{parameters[1]} and {parameters[2]} controlling for {parameters[0]}": calculate_partial_correlation(y, z, x)
    }

    return results


def calculate_all_three_parameter_correlations(data_dict):

    results = {'novice': {}, 'experienced': {}}

    parameter_combinations = [
        ('d_IP_degPFDADIBank', 'd_FM_BilleAvion', 'd_CS_rangeRudderControlPosition'),  # Turn coordination
        ('d_IP_degPFDADIAttitude', 'd_CS_rangeElevatorControlPosition', 'd_FM_ftpmAircraftVerticalSpeed'),
        # Pitch and altitude control
        ('d_ENV_ktAircraftIndicatedAirspeed', 'd_CS_rangeLeftPowerLeverPosition', 'd_FM_rpmEngine1RPM'),
        # Speed and power management
        ('d_IP_degPFDADIAttitude', 'd_FF_rangeElevatorControlForce', 'd_CS_rangeElevatorControlPosition'),
        # Elevator handling
        ('d_FC_rangeLeftFlapPosition', 'd_ENV_ktAircraftIndicatedAirspeed', 'd_FM_ftpmAircraftVerticalSpeed'),
        # Flap and speed control
    ]

    for group in ['novice', 'experienced']:
        group_data = data_dict[group]
        group_results = {}

        for i, (param1, param2, param3) in enumerate(parameter_combinations, start=1):
            parameters = [param1, param2, param3]
            correlation_result = calculate_three_parameter_correlations(group_data, parameters)
            group_results[f"Combination {i}: {param1}, {param2}, {param3}"] = correlation_result

        results[group] = group_results

    return results


def print_three_parameter_correlation_results(results):
    """
    Prints the results of three-parameter correlations for both novice and experienced groups,
    pairing each combination from the novice group with its corresponding combination from the experienced group.

    Parameters:
        results (dict): Dictionary with partial correlation results for each three-parameter combination
                        in both novice and experienced groups.
    """
    novice_results = results['novice']
    experienced_results = results['experienced']

    print("\nThree-Parameter Correlations Comparison (Novice vs. Experienced)")
    print("=" * 60)

    for combination in novice_results.keys():
        print(f"\n{combination}")

        # Print results for the Novice Group
        print("\n  Novice Group:")
        for description, correlation in novice_results[combination].items():
            print(f"    - {description}: {correlation:.3f}")

        # Print results for the Experienced Group
        print("\n  Experienced Group:")
        for description, correlation in experienced_results[combination].items():
            print(f"    - {description}: {correlation:.3f}")

        print("=" * 60)


