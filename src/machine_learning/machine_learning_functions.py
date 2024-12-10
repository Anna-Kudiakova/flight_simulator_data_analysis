import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from tsfresh import select_features

from src.utils.data_filtration import load_pilot_metadata


def calculate_frequency(values, sampling_rate):

    fft_vals = np.fft.rfft(values)
    freqs = np.fft.rfftfreq(len(values), d=1 / sampling_rate)
    # Find index of max magnitude frequency excluding the DC component
    idx = np.argmax(np.abs(fft_vals[1:])) + 1
    dominant_freq = freqs[idx]
    return dominant_freq


def calculate_rate_of_change(values, sampling_rate):
    delta_t = 1 / sampling_rate
    roc = np.mean(np.abs(np.diff(values)) / delta_t)
    return roc


def calculate_slope(values):
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    return slope


def calculate_control_reversals(values):
    return np.sum(np.diff(np.sign(np.diff(values))) != 0)


def calculate_time_in_neutral(values, threshold=0.05):
    return np.sum(np.abs(values) <= threshold) / len(values)


def calculate_control_smoothness(values, sampling_rate):

    delta_t = 1 / sampling_rate
    control_velocity = np.diff(values) / delta_t
    control_acceleration = np.diff(control_velocity) / delta_t
    control_jerk = np.diff(control_acceleration) / delta_t
    return np.mean(np.abs(control_jerk)) if len(control_jerk) > 0 else 0.0


def extract_custom_metrics(participants_data, metrics_list, sampling_rate=30):

    participant_metrics = {}
    pilot_ids = []
    flight_hours = []

    for participant, trials in participants_data.items():
        participant_metrics[participant] = {}

        first_trial_name = next(iter(trials))
        df_example = trials[first_trial_name]

        if 'Time' in df_example.columns:
            variables = df_example.columns.drop('Time')
        else:
            variables = df_example.columns

        aggregated_metrics = {metric: {var: [] for var in variables} for metric in metrics_list}

        for trial_name, df in trials.items():
            if 'Time' in df.columns:
                data = df.drop(columns='Time')
            else:
                data = df.copy()

            for var in variables:
                values = data[var].values.astype(float)

                # Check if values are valid (not empty and not all NaN)
                values_valid = (values.size > 0) and (not np.all(np.isnan(values)))

                if 'mean' in metrics_list:
                    mean_val = np.nanmean(values) if values_valid else np.nan
                    aggregated_metrics['mean'][var].append(mean_val)

                if 'std' in metrics_list:
                    std_val = np.nanstd(values) if values_valid else np.nan
                    aggregated_metrics['std'][var].append(std_val)

                if 'frequency' in metrics_list:
                    # Only calculate if valid, else np.nan
                    if values_valid:
                        freq_val = calculate_frequency(values, sampling_rate)
                    else:
                        freq_val = np.nan
                    aggregated_metrics['frequency'][var].append(freq_val)

                if 'rate_of_change' in metrics_list:
                    # Only calculate if valid, else np.nan
                    if values_valid:
                        roc_val = calculate_rate_of_change(values, sampling_rate)
                    else:
                        roc_val = np.nan
                    aggregated_metrics['rate_of_change'][var].append(roc_val)

                if 'slope' in metrics_list:
                    # Only calculate if valid, else np.nan
                    if values_valid:
                        slope_val = calculate_slope(values)
                    else:
                        slope_val = np.nan
                    aggregated_metrics['slope'][var].append(slope_val)

        for metric in metrics_list:
            for var in variables:
                mean_val = np.mean(aggregated_metrics[metric][var])
                participant_metrics[participant][f"{var}_{metric}"] = mean_val

        pilot_metadata = load_pilot_metadata()

        flight_hour = pilot_metadata.loc[pilot_metadata['Subject'] == participant, 'Hours'].values[0]
        flight_hours.append(flight_hour)
        pilot_ids.append(participant)

    x = pd.DataFrame.from_dict(participant_metrics, orient='index')
    y = pd.Series(flight_hours, index=pilot_ids, name='flight_hours')

    return x, y


def extract_statistical_features(participants_data):
    feature_list = []
    pilot_ids = []
    flight_hours = []

    for pilot, trials in participants_data.items():
        trial_features = []
        for trial_name, df in trials.items():
            if 'Time' in df.columns:
                data = df.drop(['Time'], axis=1)
            else:
                data = df.copy()

            stats = data.agg(['mean', 'std', 'min', 'max', 'median']).to_numpy().flatten()
            trial_features.append(stats)

        trial_features = np.array(trial_features)
        aggregated_features = np.mean(trial_features, axis=0)

        feature_list.append(aggregated_features)
        pilot_ids.append(pilot)

        pilot_metadata = load_pilot_metadata()
        flight_hour = pilot_metadata.loc[pilot_metadata['Subject'] == pilot, 'Hours'].values[0]
        flight_hours.append(flight_hour)

    feature_columns = []

    # extract column names
    first_pilot_id = pilot_ids[0]
    first_trial_name = next(iter(participants_data[first_pilot_id]))
    first_trial_df = participants_data[first_pilot_id][first_trial_name]
    variables = first_trial_df.columns.drop('Time')
    for stat in ['mean', 'std', 'min', 'max', 'median']:
        for var in variables:  # Skip 'Time'
            feature_columns.append(f"{var}_{stat}")

    x = pd.DataFrame(feature_list, columns=feature_columns)
    y = pd.Series(flight_hours, name='flight_hours', index=pilot_ids)

    return x, y


def encode_labels(y):
    # Define bins according to the classification criteria
    bins = [-np.inf, 40, 140, 240, np.inf]
    labels = ['Novice', 'Private', 'Intermediate', 'Commercial']

    # Classify based on the provided bins and labels
    y_cat = pd.cut(y, bins=bins, labels=labels, right=False)
    return y_cat


def define_pca_components_number(x, accuracy):
    pca_full = PCA().fit(x)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = np.argmax(cumulative_variance >= accuracy) + 1
    print(f"Number of components to explain 95% variance: {n_components_95}")
    print(f"Validation")
    pca = PCA(n_components=n_components_95, random_state=42)
    pca.fit_transform(x)
    print("Explained variance ratio for selected components:")
    print(pca.explained_variance_ratio_)
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2f}")
    return n_components_95

def hyperparameter_tuning(X, y):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(clf, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                        scoring='accuracy')
    grid.fit(X, y)
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validated accuracy: {grid.best_score_:.2f}")
    return grid.best_estimator_

def train_evaluate_classifier(X, y):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"Logistic Regression Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
    return scores


def prepare_tsfresh_data(participants_data):
    records = []
    for pilot, trials in participants_data.items():
        for trial_name, df in trials.items():
            # Add identifiers for tsfresh
            df = df.copy()
            df['pilot'] = pilot
            df['trial'] = trial_name
            records.append(df)

    df_all = pd.concat(records, ignore_index=True)
    return df_all

def select_relevant_features(extracted_features, y):
    # Select features relevant to flight hours
    relevant_features = select_features(extracted_features, y)
    print(f"Selected features shape: {relevant_features.shape}")
    return relevant_features