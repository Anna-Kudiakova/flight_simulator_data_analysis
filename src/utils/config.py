import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "trials")
COLUMNS_DIR = os.path.join(BASE_DIR, "parameters", "columns.csv")
COLUMNS_DIR_EXTENDED = os.path.join(BASE_DIR, "parameters", "extended_columns.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_trials", "processed_trials.pkl")
PROCESSED_DATA_DIR_EXTENDED = os.path.join(BASE_DIR, "processed_trials", "processed_trials_extended.pkl")
TRIAL_CONDITION_DIR = os.path.join(BASE_DIR, "trial_conditions", "trial_info.csv")
PILOT_METADATA_DIR = os.path.join(BASE_DIR, "pilot_info", "pilot_metadata.csv")
Y_TSFRESH_DIR = os.path.join(BASE_DIR, "tsfresh_data", "y_tsfresh.pkl")
X_TSFRESH_DIR = os.path.join(BASE_DIR, "tsfresh_data", "x_tsfresh.pkl")
Y_CUSTOM_DIR = os.path.join(BASE_DIR, "tsfresh_data", "y_custom.pkl")
X_CUSTOM_DIR = os.path.join(BASE_DIR, "tsfresh_data", "x_custom.pkl")
Y_STAT_DIR = os.path.join(BASE_DIR, "tsfresh_data", "y_stat.pkl")
X_STAT_DIR = os.path.join(BASE_DIR, "tsfresh_data", "x_stat.pkl")

CSV_DELIMITER = ";"
