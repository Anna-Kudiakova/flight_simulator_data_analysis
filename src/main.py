from src.data_analysis.analysis_functions import perform_t_test, calculate_distribution, \
    calculate_metrics, structure_and_normalize_participant_data, visualize_distribution_after_normalization, \
    structure_participant_data
from src.data_visualization.visualization_functions import visualize_mann_whitney_u_tests, save_all_results
from src.utils.config import COLUMNS_DIR, PROCESSED_DATA_DIR
from src.utils.data_filtration import load_trial_conditions, filter_trial_conditions, get_filtered_participant_data, \
    load_pilot_metadata, categorize_pilots
from src.utils.data_loader import load_all_data, load_or_process_data, get_headers


def main():
    data_dir = PROCESSED_DATA_DIR
    columns_dir = COLUMNS_DIR
    participants_data = load_or_process_data(data_dir, columns_dir)

    trial_conditions = load_trial_conditions()
    trials_easy_conditions = filter_trial_conditions(trial_conditions, 'NO WIND', 'SUCCESSFUL')
    participants_data_easy_conditions = get_filtered_participant_data(participants_data, trials_easy_conditions)

    # print(f"Participant number: {len(participants_data_easy_conditions)}")
    # for participant, sss in participants_data_easy_conditions.items():
    #     print(f"Participant: {participant}")
    #     print(f"Number of sss: {len(sss)}")
    #     for trial_name, df in sss.items():
    #         print(f"  Trial: {trial_name}")
    #         print(df.head())

    pilot_metadata = load_pilot_metadata()

    novice_pilots, experienced_pilots = categorize_pilots(pilot_metadata)

    print(f"Number of novice pilots: {len(novice_pilots)}")
    print(f"Number of experienced pilots: {len(experienced_pilots)}")

    all_columns = get_headers(COLUMNS_DIR)

    # for column in extended_columns.csv:
    #     calculate_distribution(participants_data_easy_conditions, column, novice_pilots, experienced_pilots)

    parameters_dict = {
        "elevator_force": {
            "data": "d_FF_rangeElevatorControlForce",
            "normalization": "yeo-johnson",
            "units": "Newtons",
            "metrics": {
                "mean": "Newtons",
                "std": "Newtons",
                "coefficient_of_variation": "Ratio: Std/Mean",
                "rate_of_change": "Newtons/second",
            },
            "viz_rows": 2,
            "viz_cols": 2,
        },
        "aileron_force": {
            "data": "d_FF_rangeAileronControlForce",
            "normalization": "z-score",
            "units": "Newtons",
            "metrics": {
                "mean": "Newtons",
                "std": "Newtons",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "Newtons/second",
            },
            "viz_rows": 2,
            "viz_cols": 2,
        },
        "rudder_force": {
            "data": "d_FF_rangeRudderControlForce",
            "normalization": "yeo-johnson",
            "units": "Newtons",
            "metrics": {
                "mean": "Newtons",
                "std": "Newtons",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "Newtons/second",
            },
            "viz_rows": 2,
            "viz_cols": 2,
        },
        "elevator_position": {
            "data": "d_CS_rangeElevatorControlPosition",
            "normalization": "yeo-johnson",
            "units": "Degrees (°)",
            "metrics": {
                "mean": "Degrees (°)",
                "std": "Degrees (°)",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "Degrees/second",
                "control_reversals": "Count",
                "time_in_neutral": "Proportion of time in neutral",
                "control_smoothness": "Degrees/second³",
            },
            "viz_rows": 3,
            "viz_cols": 3,
        },
        "aileron_position": {
            "data": "d_CS_rangeAileronControlPosition",
            "normalization": "yeo-johnson",
            "units": "Degrees (°)",
            "metrics": {
                "mean": "Degrees (°)",
                "std": "Degrees (°)",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "Degrees/second",
                "control_reversals": "Count",
                "time_in_neutral": "Proportion of time in neutral",
                "control_smoothness": "Degrees/second³",
            },
            "viz_rows": 3,
            "viz_cols": 3,
        },
        "rudder_position": {
            "data": "d_CS_rangeRudderControlPosition",
            "normalization": "yeo-johnson",
            "units": "Degrees (°)",
            "metrics": {
                "mean": "Degrees (°)",
                "std": "Degrees (°)",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "Degrees/second",
                "control_reversals": "Count",
                "time_in_neutral": "Proportion of time in neutral",
                "control_smoothness": "Degrees/second³",
            },
            "viz_rows": 3,
            "viz_cols": 3,
        },
        "throttle_position": {
            "data": "d_CS_rangeLeftPowerLeverPosition",
            "normalization": "box-cox",
            "units": "Percentage (%)",
            "metrics": {
                "mean": "Percentage (%)",
                "std": "Percentage (%)",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "Percentage/second",
                "control_reversals": "Count",
                "time_in_neutral": "Proportion of time in neutral",
                "control_smoothness": "Percentage/second³",
            },
            "viz_rows": 3,
            "viz_cols": 3,
        },
        "trim_position": {
            "data": "d_FC_rangeElevatorTrimPosition",
            "normalization": "yeo-johnson",
            "units": "Degrees (°)",
            "metrics": {
                "mean": "Degrees (°)",
                "std": "Degrees (°)",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "Degrees/second",
                "control_reversals": "Count",
                "time_in_neutral": "Proportion",
                "control_smoothness": "Degrees/second³",
            },
            "viz_rows": 3,
            "viz_cols": 3,
        },
        "pitch_state": {
            "data": "d_IP_degPFDADIAttitude",
            "normalization": "yeo-johnson",
            "units": "Degrees (°)",
            "metrics": {
                "mean": "Degrees (°)",
                "std": "Degrees (°)",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "Degrees/second",
            },
            "viz_rows": 2,
            "viz_cols": 2,
        },
        "bank_state": {
            "data": "d_IP_degPFDADIBank",
            "normalization": "yeo-johnson",
            "units": "Degrees (°)",
            "metrics": {
                "mean": "Degrees (°)",
                "std": "Degrees (°)",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "Degrees/second",
            },
            "viz_rows": 2,
            "viz_cols": 2,
        },
        "airspeed_state": {
            "data": "d_ENV_ktAircraftIndicatedAirspeed",
            "normalization": "box-cox",
            "units": "Knots (kt)",
            "metrics": {
                "mean": "Knots (kt)",
                "std": "Knots (kt)",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "Knots/second",
            },
            "viz_rows": 2,
            "viz_cols": 2,
        },
        "vertical_speed_state": {
            "data": "d_FM_ftpmAircraftVerticalSpeed",
            "normalization": "yeo-johnson",
            "units": "ft/min (Feet per minute)",
            "metrics": {
                "mean": "ft/min",
                "std": "ft/min",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "(ft/min)/second",
            },
            "viz_rows": 2,
            "viz_cols": 2,
        },
        "flaps_state": {
            "data": "d_FC_rangeLeftFlapPosition",
            "normalization": "no",
            "units": "Position setting",
            "metrics": {
                "mean": "Position setting",
                "std": "Position setting",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "Position setting/second",
                "control_reversals": "Count",
                "control_smoothness": "Position setting/second³",
                "time_in_neutral": "Proportion",
            },
            "viz_rows": 3,
            "viz_cols": 3,
        },
        "power_state": {
            "data": "d_FM_rpmEngine1RPM",
            "normalization": "box-cox",
            "units": "RPM (Revolutions per minute)",
            "metrics": {
                "mean": "RPM",
                "std": "RPM",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "RPM/second",
            },
            "viz_rows": 2,
            "viz_cols": 2,
        },
        "heading_state": {
            "data": "d_FM_degAircraftTrueHeading",
            "normalization": "no",
            "units": "Degrees (°)",
            "metrics": {
                "mean": "Degrees (°)",
                "std": "Degrees (°)",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "Degrees/second",
            },
            "viz_rows": 2,
            "viz_cols": 2,
        },
        "slip_indicator_state": {
            "data": "d_FM_BilleAvion",
            "normalization": "yeo-johnson",
            "units": "Degrees (°)",
            "metrics": {
                "mean": "Degrees (°)",
                "std": "Degrees (°)",
                "coefficient_of_variation": "Ratio",
                "rate_of_change": "Degrees/second",
                "slip_duration": "Proportion of time outside of neutral state",
            },
            "viz_rows": 2,
            "viz_cols": 3,
        }
    }

    all_metrics = {}

    for column_name, column_info in parameters_dict.items():
        curr_column = column_info["data"]
        normalization_type = column_info["normalization"]
        curr_metrics_dict = column_info["metrics"]

        novice_data, experienced_data = structure_participant_data(
            participants_data_easy_conditions,
            novice_pilots, experienced_pilots,
            curr_column)

        # normalized_novice_data, normalized_experienced_data = structure_and_normalize_participant_data(
        #     participants_data_easy_conditions,
        #     novice_pilots, experienced_pilots,
        #     curr_column,
        #     normalization_type)
        #
        # visualize_distribution_after_normalization(normalized_novice_data,
        #                                            normalized_experienced_data,
        #                                            curr_column)

        novice_metrics = calculate_metrics(novice_data, metrics_list=curr_metrics_dict.keys())
        experienced_metrics = calculate_metrics(experienced_data, metrics_list=curr_metrics_dict.keys())

        all_metrics[column_name] = {
            'novice': {pilot: metrics for pilot, metrics in novice_metrics.items()},
            'experienced': {pilot: metrics for pilot, metrics in experienced_metrics.items()}
        }

    for column_name, column_info in parameters_dict.items():
        novice_column_metrics = all_metrics[column_name]['novice']
        experienced_column_metrics = all_metrics[column_name]['experienced']

        # Visualize selected metrics for each force type
        visualize_mann_whitney_u_tests(novice_column_metrics,
                                       experienced_column_metrics,
                                       column_info["metrics"],
                                       column_info["viz_rows"],
                                       column_info["viz_cols"],
                                       column_name)

    save_all_results()


if __name__ == "__main__":
    main()
