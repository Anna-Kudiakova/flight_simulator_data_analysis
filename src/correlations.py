from src.data_analysis.correlation_functions import prepare_data_for_correlation, calculate_two_parameter_correlations, \
    print_significant_correlations, calculate_all_three_parameter_correlations, \
    print_three_parameter_correlation_results
from src.utils.data_filtration import load_trial_conditions, filter_trial_conditions, get_filtered_participant_data, \
    load_pilot_metadata, categorize_pilots
from src.utils.data_loader import load_or_process_data


def correlations():
    participants_data = load_or_process_data(selected_columns=True)

    trial_conditions = load_trial_conditions()
    trials_easy_conditions = filter_trial_conditions(trial_conditions, 'NO WIND', 'SUCCESSFUL')
    participants_data_easy_conditions = get_filtered_participant_data(participants_data, trials_easy_conditions)
    pilot_metadata = load_pilot_metadata()

    novice_pilots, experienced_pilots = categorize_pilots(pilot_metadata)

    prepared_data = prepare_data_for_correlation(participants_data_easy_conditions, novice_pilots, experienced_pilots)

    two_parameter_correlations = calculate_two_parameter_correlations(prepared_data, method='spearman')

    print_significant_correlations(two_parameter_correlations)

    three_parameter_correlations = calculate_all_three_parameter_correlations(prepared_data)
    print_three_parameter_correlation_results(three_parameter_correlations)


if __name__ == '__main__':
    correlations()
