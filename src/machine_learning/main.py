import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.machine_learning.machine_learning_functions import extract_statistical_features, encode_labels, \
    define_pca_components_number, hyperparameter_tuning, train_evaluate_classifier, extract_custom_metrics
from src.utils.config import PROCESSED_DATA_DIR_EXTENDED, COLUMNS_DIR_EXTENDED
from src.utils.data_loader import load_or_process_data


def main():
    data_dir = PROCESSED_DATA_DIR_EXTENDED
    columns_dir = COLUMNS_DIR_EXTENDED
    participants_data = load_or_process_data(data_dir, columns_dir)

    # Default statistics
    x_stat, y_stat = extract_statistical_features(participants_data)

    # Custom statistics
    # metrics_list = [
    #     'mean', 'std', 'coefficient_of_variation', 'frequency', 'rate_of_change',
    #     'slope'
    # ]
    # Custom statistics
    # x_stat, y_stat = extract_custom_metrics(participants_data, metrics_list)

    y_stat_cat = encode_labels(y_stat)
    scaler = StandardScaler()
    x_stat_scaled = scaler.fit_transform(x_stat)

    # Step 1: Impute missing values
    imputer = SimpleImputer(strategy='mean')
    x_stat_scaled = imputer.fit_transform(x_stat_scaled)

    n_components_98 = define_pca_components_number(x_stat_scaled, 0.98)
    # n_components_98 = 23
    pca = PCA(n_components=23, random_state=42)
    x_stat_pca = pca.fit_transform(x_stat_scaled)
    best_clf_stat = hyperparameter_tuning(x_stat_pca, y_stat_cat)
    # accuracy_stat = train_evaluate_classifier(x_stat_pca, y_stat_cat)


if __name__ == "__main__":
    main()
