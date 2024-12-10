from src.data_analysis.analysis_functions import structure_participant_data, find_median_trial_across_group
from src.data_visualization.visualization_functions import plot_time_series_for_median_trials
from src.utils.data_filtration import load_trial_conditions, filter_trial_conditions, get_filtered_participant_data, \
    load_pilot_metadata, categorize_pilots
from src.utils.data_loader import load_all_data, load_or_process_data, get_headers


def visualize_timeseries():
    participants_data = load_or_process_data()

    trial_conditions = load_trial_conditions()
    trials_easy_conditions = filter_trial_conditions(trial_conditions, 'NO WIND', 'SUCCESSFUL')
    participants_data_easy_conditions = get_filtered_participant_data(participants_data, trials_easy_conditions)
    pilot_metadata = load_pilot_metadata()
    novice_pilots, experienced_pilots = categorize_pilots(pilot_metadata)

    elevator_position_closest_trials = find_median_trial_across_group(
        participants_data_easy_conditions,
        novice_pilots,
        experienced_pilots,
        'd_CS_rangeElevatorControlPosition')
    plot_time_series_for_median_trials(
        participants_data_easy_conditions,
        elevator_position_closest_trials,
        'd_CS_rangeElevatorControlPosition',
        'Elevator Position',
        'Degrees (°)',
        sampling_rate=30)

    throttle_position_closest_trials = find_median_trial_across_group(
        participants_data_easy_conditions,
        novice_pilots,
        experienced_pilots,
        'd_CS_rangeLeftPowerLeverPosition')
    plot_time_series_for_median_trials(
        participants_data_easy_conditions,
        throttle_position_closest_trials,
        'd_CS_rangeLeftPowerLeverPosition',
        'Throttle Position',
        'Percentage (%)',
        sampling_rate=30)


if __name__ == '__main__':
    visualize_timeseries()