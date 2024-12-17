import time

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, ParameterGrid, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from tsfresh import select_features
from xgboost import XGBClassifier, DMatrix, cv, train

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
    feature_list = []
    pilot_ids = []
    flight_hours = []

    for participant, trials in participants_data.items():
        trial_metrics = []

        first_trial_name = next(iter(trials))
        df_example = trials[first_trial_name]

        if 'Time' in df_example.columns:
            variables = df_example.columns.drop('Time')
        else:
            variables = df_example.columns

        # Initialize metrics dictionary
        aggregated_metrics = {metric: {var: [] for var in variables} for metric in metrics_list}

        for trial_name, df in trials.items():
            if 'Time' in df.columns:
                data = df.drop(columns='Time')
            else:
                data = df.copy()

            for var in variables:
                values = data[var].values.astype(float)

                # Check if values are valid
                values_valid = (values.size > 0) and (not np.all(np.isnan(values)))

                if not values_valid:
                    continue

                # Compute metrics based on the list provided
                if 'mean' in metrics_list:
                    aggregated_metrics['mean'][var].append(np.nanmean(values))
                if 'std' in metrics_list:
                    aggregated_metrics['std'][var].append(np.nanstd(values))
                if 'frequency' in metrics_list:
                    aggregated_metrics['frequency'][var].append(calculate_frequency(values, sampling_rate))
                if 'rate_of_change' in metrics_list:
                    aggregated_metrics['rate_of_change'][var].append(calculate_rate_of_change(values, sampling_rate))
                if 'control_reversals' in metrics_list and 'Position' in var:
                    aggregated_metrics['control_reversals'][var].append(calculate_control_reversals(values))
                if 'control_smoothness' in metrics_list and 'Position' in var:
                    aggregated_metrics['control_smoothness'][var].append(calculate_control_smoothness(values, sampling_rate))
                if 'skewness' in metrics_list:
                    if np.std(values[~np.isnan(values)]) > 1e-8:
                        aggregated_metrics['skewness'][var].append(skew(values[~np.isnan(values)]))
                    else:
                        aggregated_metrics['skewness'][var].append(0)
                if 'kurtosis' in metrics_list:
                    if np.std(values[~np.isnan(values)]) > 1e-8:
                        aggregated_metrics['kurtosis'][var].append(kurtosis(values[~np.isnan(values)], fisher=True))
                    else:
                        aggregated_metrics['kurtosis'][var].append(0)
                if 'slope' in metrics_list:
                    aggregated_metrics['slope'][var].append(calculate_slope(values[~np.isnan(values)]))

        # Aggregate metrics across all trials for this participant
        participant_features = []
        for metric in metrics_list:
            for var in variables:
                if metric in ['control_reversals', 'control_smoothness'] and 'Position' not in var:
                    continue
                elif len(aggregated_metrics[metric][var]) > 0:
                    mean_val = np.nanmean(aggregated_metrics[metric][var])
                else:
                    mean_val = np.nan
                participant_features.append(mean_val)

        feature_list.append(participant_features)
        pilot_ids.append(participant)

        pilot_metadata = load_pilot_metadata()
        flight_hour = pilot_metadata.loc[pilot_metadata['Subject'] == participant, 'Hours'].values[0]
        flight_hours.append(flight_hour)

    # Generate feature column names
    feature_columns = []
    for metric in metrics_list:
        for var in variables:
            if metric in ['control_reversals', 'control_smoothness'] and 'Position' not in var:
                # Skip metrics requiring 'Position' for variables that don't have it
                continue
            feature_columns.append(f"{var}_{metric}")

    x = pd.DataFrame(feature_list, columns=feature_columns)
    y = pd.Series(flight_hours, name='flight_hours', index=pilot_ids)

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
    print(f"Number of components to explain {accuracy * 100}% variance: {n_components_95}")
    return n_components_95


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


def custom_split(x_data, y_data, split_number, n_test_samples=2, xgboost_model=False):
    np.random.seed(42 + split_number)

    # Combine features and labels for processing
    data = pd.DataFrame(x_data)
    if xgboost_model:
        data['label'] = y_data
    else:
        data['label'] = y_data.values


    test_indices = []
    train_indices = []

    if xgboost_model:
        y_data_array = pd.Series(y_data).unique()
    else:
        y_data_array = y_data.unique()

    for category in y_data_array:
        # Get all indices for the current category
        category_indices = data[data['label'] == category].index.tolist()

        # Shuffle indices and select test samples
        np.random.shuffle(category_indices)
        selected_test = category_indices[:n_test_samples]
        remaining_train = category_indices[n_test_samples:]

        test_indices.extend(selected_test)
        train_indices.extend(remaining_train)

    x_data = pd.DataFrame(x_data)
    y_data = pd.Series(y_data)

    x_train = x_data.iloc[train_indices]
    x_test = x_data.iloc[test_indices]
    y_train = y_data.iloc[train_indices]
    y_test = y_data.iloc[test_indices]

    return x_train, x_test, y_train, y_test


def custom_cross_validation(x_data, y_data, model, n_splits=10, n_test_samples=2, pca_variance=0.98):
    accuracies = []
    all_y_test = []
    all_y_test_pred = []

    for split in range(n_splits):
        # Custom split for this fold
        x_train, x_test, y_train, y_test = custom_split(x_data, y_data, split, n_test_samples=n_test_samples)

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        x_train = imputer.fit_transform(x_train)
        x_test = imputer.transform(x_test)

        # Check for invalid values after imputation
        x_train = np.nan_to_num(x_train, nan=0.0, posinf=np.max(x_train[np.isfinite(x_train)]), neginf=np.min(x_train[np.isfinite(x_train)]))
        x_test = np.nan_to_num(x_test, nan=0.0, posinf=np.max(x_test[np.isfinite(x_test)]), neginf=np.min(x_test[np.isfinite(x_test)]))


        # Standardize features
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Apply PCA
        n_components = define_pca_components_number(x_train_scaled, pca_variance)
        pca = PCA(n_components=n_components, random_state=42)
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_test_pca = pca.transform(x_test_scaled)

        assert len(x_train) == len(y_train), "Mismatch in train features and labels"
        assert len(x_test) == len(y_test), "Mismatch in test features and labels"

        # float32_max = np.finfo(np.float32).max
        # float32_min = np.finfo(np.float32).min
        #
        # # Convert all values greater than float32_max or less than float32_min
        # x_train_pca = np.clip(x_train_pca, float32_min, float32_max)
        # x_test_pca = np.clip(x_test_pca, float32_min, float32_max)


        # Initialize and train logistic regression with GridSearchCV
        clf = clone(model)
        clf.fit(x_train_pca, y_train)  # Train the model on the training set

        # Predict and evaluate
        y_test_pred = clf.predict(x_test_pca)
        accuracy = accuracy_score(y_test, y_test_pred)
        accuracies.append(accuracy)

        # Store predictions for final report
        all_y_test.extend(y_test)
        all_y_test_pred.extend(y_test_pred)


        # Print the best parameters and performance for the current fold
        print(f"Fold {split + 1} Accuracy: {accuracy:.2f}")
        print(f"Best parameters for this fold: {clf.best_params_}")
        print("Classification Report:")
        print(classification_report(y_test, y_test_pred, zero_division=0))

    # Average accuracy across all folds
    average_accuracy = np.mean(accuracies)
    print(f"Average Test Accuracy across {n_splits} folds: {average_accuracy:.2f}")
    # Final report based on all folds
    print("\nFinal Classification Report (Aggregated over all folds):")
    report(all_y_test, all_y_test_pred)
    class_labels = ['Novice', 'Intermediate', 'Private', 'Commercial']

    # Generate the confusion matrix
    cm = confusion_matrix(all_y_test, all_y_test_pred, labels=class_labels)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap="Blues")
    plt.title("Corrected Aggregated Confusion Matrix Across All Folds")
    plt.show()

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Novice', 'Intermediate', 'Private', 'Commercial'])
    disp.plot(cmap=plt.cm.Blues)  # You can change the colormap as needed
    plt.title("Aggregated Confusion Matrix Across All Folds")
    plt.show()


    return average_accuracy

def custom_cross_validation_xgboost(x_data, y_data, n_splits=10, n_test_samples=2, pca_variance=0.98):
    accuracies = []
    all_y_test = []
    all_y_test_pred = []

    # Encode labels for XGBoost
    label_encoder = LabelEncoder()
    y_data = label_encoder.fit_transform(y_data)

    param_grid = {
        'max_depth': [3],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [1.0],
        'gamma': [0, 0.1],
        'min_child_weight': [1],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1],
    }


    for split in range(n_splits):
        # Custom split for this fold
        x_train, x_test, y_train, y_test = custom_split(x_data, y_data, split, n_test_samples=n_test_samples,
                                                        xgboost_model=True)

        # Split training set into train and validation subsets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        # Preprocess data as before
        imputer = SimpleImputer(strategy='mean')
        x_train = imputer.fit_transform(x_train)
        x_val = imputer.transform(x_val)
        x_test = imputer.transform(x_test)

        x_train = np.nan_to_num(x_train, nan=0.0, posinf=np.max(x_train[np.isfinite(x_train)]), neginf=np.min(x_train[np.isfinite(x_train)]))
        x_val = np.nan_to_num(x_val, nan=0.0, posinf=np.max(x_val[np.isfinite(x_val)]), neginf=np.min(x_val[np.isfinite(x_val)]))
        x_test = np.nan_to_num(x_test, nan=0.0, posinf=np.max(x_test[np.isfinite(x_test)]), neginf=np.min(x_test[np.isfinite(x_test)]))


        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)

        n_components = define_pca_components_number(x_train_scaled, pca_variance)
        pca = PCA(n_components=n_components, random_state=42)
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_val_pca = pca.transform(x_val_scaled)
        x_test_pca = pca.transform(x_test_scaled)

        # Grid Search using validation data
        best_params = None
        best_val_accuracy = -1
        for params in ParameterGrid(param_grid):
            dtrain = DMatrix(x_train_pca, label=y_train, missing=np.inf)
            dval = DMatrix(x_val_pca, label=y_val, missing=np.inf)

            params['objective'] = 'multi:softmax'
            params['num_class'] = len(np.unique(y_train))
            params['eval_metric'] = 'mlogloss'

            model = train(params, dtrain, num_boost_round=100)
            y_val_pred = model.predict(dval)
            y_val_pred = np.round(y_val_pred)

            val_accuracy = accuracy_score(y_val, y_val_pred)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = params

        # Evaluate on test set
        dtest = DMatrix(x_test_pca, label=y_test, missing=np.inf)
        final_model = train(best_params, dtrain, num_boost_round=100)
        y_test_pred = final_model.predict(dtest)
        test_accuracy = accuracy_score(y_test, np.round(y_test_pred))
        accuracies.append(test_accuracy)

        # Decode numerical labels back to string categories
        y_test_decoded = label_encoder.inverse_transform(y_test.astype(int))
        y_test_pred_decoded = label_encoder.inverse_transform(y_test_pred.astype(int))

        # Store decoded predictions for final report
        all_y_test.extend(y_test_decoded)
        all_y_test_pred.extend(y_test_pred_decoded)

        # Print results
        print(f"Fold {split + 1} Validation Accuracy: {best_val_accuracy:.2f}")
        print(f"Fold {split + 1} Test Accuracy: {test_accuracy:.2f}")
        print(f"Best Parameters for this fold: {best_params}")
        print("Classification Report:")
        print(classification_report(y_test, np.round(y_test_pred), zero_division=0))

    # Average accuracy across all folds
    average_accuracy = np.mean(accuracies)
    print(f"Average Test Accuracy across {n_splits} folds: {average_accuracy:.2f}")
    # Final report based on all folds
    print("\nFinal Classification Report (Aggregated over all folds):")
    report(all_y_test, all_y_test_pred)


custom_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def logistic_regression():
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga']
    }
    base_model = LogisticRegression(max_iter=10000, class_weight='balanced', random_state=42)
    clf = GridSearchCV(base_model, param_grid, cv=custom_cv, scoring='accuracy', n_jobs=-1, refit=True)
    return clf


def random_forest_model():
    param_grid = {
        'n_estimators': [25, 50, 100],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 4, 8],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    base_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    clf = GridSearchCV(base_model, param_grid, cv=custom_cv, scoring='accuracy', n_jobs=-1, refit=True)
    return clf


def xgboost_model():
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.01],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    base_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
    clf = GridSearchCV(base_model, param_grid, cv=custom_cv, scoring='accuracy', n_jobs=-1, refit=True)
    return base_model


def svm_model():
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    }
    base_model = SVC(class_weight='balanced', probability=True, random_state=42)
    clf = GridSearchCV(base_model, param_grid, cv=custom_cv, scoring='accuracy', n_jobs=-1, refit=True)
    return clf


def mlp_model():
    param_grid = {
        'hidden_layer_sizes': [
            (5,), (10,), (20,), (30,), (40,),
            (10, 10), (20, 10), (30, 10), (20, 20),
            (10, 10, 10), (20, 10, 10), (30, 20, 10), (20, 10, 5), (15, 10, 5)

        ],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'batch_size': [4, 8, 16],
        'learning_rate_init': [0.001, 0.01, 0.1],  # Initial learning rates
        'learning_rate': ['constant'],# Learning rate strategies , delete adaptive
    }
    base_model = MLPClassifier(max_iter=10000, random_state=42)
    clf = GridSearchCV(base_model, param_grid, cv=custom_cv, scoring='accuracy', n_jobs=-1, refit=True)
    return clf


def report(y_true, y_pred):
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    report_df = pd.DataFrame(report).transpose()

    # Exclude "accuracy", "macro avg", and "weighted avg" rows
    class_metrics = report_df.iloc[:-3][['precision', 'recall', 'f1-score']]

    # Add "difficulty_score" as 1 - F1-score (lower F1 = higher difficulty)
    class_metrics['difficulty_score'] = 1 - class_metrics['f1-score']

    # Rank classes by difficulty (ascending F1-score)
    class_metrics_sorted = class_metrics.sort_values(by='difficulty_score', ascending=False)

    # Print the ranked difficulty with complementing the classification report
    print("\nClass Difficulty Ranking (Most Difficult to Easiest):")
    for idx, row in class_metrics_sorted.iterrows():
        print(f"Class {idx}: "
              f"Precision={row['precision']:.2f}, Recall={row['recall']:.2f}, "
              f"F1-Score={row['f1-score']:.2f}, Difficulty Score={row['difficulty_score']:.2f}")
