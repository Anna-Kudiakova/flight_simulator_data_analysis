import os
import pickle

from src.machine_learning.machine_learning_functions import extract_statistical_features, encode_labels, \
    define_pca_components_number, logistic_regression, train_evaluate_classifier, extract_custom_metrics, custom_split, \
    custom_cross_validation, mlp_model, svm_model, xgboost_model, random_forest_model, custom_cross_validation_xgboost
from src.utils.config import PROCESSED_DATA_DIR_EXTENDED, COLUMNS_DIR_EXTENDED, Y_CUSTOM_DIR, X_CUSTOM_DIR, X_STAT_DIR, \
    Y_STAT_DIR
from src.utils.data_loader import load_or_process_data


def main():
    data_dir = PROCESSED_DATA_DIR_EXTENDED
    columns_dir = COLUMNS_DIR_EXTENDED
    x_stat_path = X_STAT_DIR
    y_stat_path = Y_STAT_DIR

    if os.path.exists(x_stat_path) and os.path.exists(y_stat_path):
        print("Loading precomputed data...")
        with open(x_stat_path, "rb") as f:
            x_stat = pickle.load(f)
        with open(y_stat_path, "rb") as f:
            y_stat = pickle.load(f)
    else:
        os.makedirs(os.path.dirname(x_stat_path), exist_ok=True)
        os.makedirs(os.path.dirname(y_stat_path), exist_ok=True)

        participants_data = load_or_process_data(data_dir, columns_dir)

        x_stat, y_stat = extract_statistical_features(participants_data)

        with open(x_stat_path, "wb") as f:
            pickle.dump(x_stat, f)
        with open(y_stat_path, "wb") as f:
            pickle.dump(y_stat, f)

        print("Data saved successfully.")

    y_stat_cat = encode_labels(y_stat)

    # Try different models
    # model = logistic_regression()
    # model = random_forest_model()
    model = svm_model()
    # model = mlp_model()

    custom_cross_validation(x_stat, y_stat_cat, n_test_samples=2, model=model, pca_variance=0.98)
    # custom_cross_validation_xgboost(x_stat, y_stat_cat, pca_variance=0.8)


if __name__ == "__main__":
    main()