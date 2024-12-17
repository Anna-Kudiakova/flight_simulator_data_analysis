import os
import pickle

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

from src.machine_learning.machine_learning_functions import prepare_tsfresh_data, encode_labels, \
    select_relevant_features, define_pca_components_number, logistic_regression, custom_split, custom_cross_validation, \
    xgboost_model, svm_model, mlp_model, random_forest_model, custom_cross_validation_xgboost
from src.utils.config import PROCESSED_DATA_DIR_EXTENDED, COLUMNS_DIR_EXTENDED, X_TSFRESH_DIR, \
    Y_TSFRESH_DIR
from src.utils.data_filtration import load_pilot_metadata
from src.utils.data_loader import load_or_process_data


def main():
    data_dir = PROCESSED_DATA_DIR_EXTENDED
    columns_dir = COLUMNS_DIR_EXTENDED
    x_tsfresh_path = X_TSFRESH_DIR
    y_tsfresh_path = Y_TSFRESH_DIR

    if os.path.exists(x_tsfresh_path) and os.path.exists(y_tsfresh_path):
        print("Loading precomputed data...")
        with open(x_tsfresh_path, "rb") as f:
            x_tsfresh = pickle.load(f)
        with open(y_tsfresh_path, "rb") as f:
            y_tsfresh = pickle.load(f)
            y_tsfresh_cat = encode_labels(y_tsfresh)
    else:
        print("Calculating tsfresh data...")
        participants_data = load_or_process_data(data_dir, columns_dir)

        df_tsfresh = prepare_tsfresh_data(participants_data)

        df_melted = df_tsfresh.melt(id_vars=['Time', 'pilot', 'trial'],
                                    var_name='variable',
                                    value_name='value')

        df_melted.dropna(subset=['value'], inplace=True)

        print("Extracting features with tsfresh...")

        settings = EfficientFCParameters()
        extracted_features = extract_features(
            df_melted,
            column_id='pilot',
            column_sort='Time',
            column_kind='variable',
            column_value='value',
            default_fc_parameters=settings,
        )
        extracted_features.fillna(0, inplace=True)  # Simple imputation

        print(f"Extracted features shape: {extracted_features.shape}")
        pilot_metadata = load_pilot_metadata()

        # Prepare labels for tsfresh (flight hours per pilot)
        y_tsfresh = pilot_metadata.set_index('Subject').loc[extracted_features.index, 'Hours']

        y_tsfresh_cat = encode_labels(y_tsfresh)
        y_tsfresh_num = y_tsfresh_cat.cat.codes

        relevant_features_tsfresh = select_relevant_features(extracted_features, y_tsfresh_num)

        x_tsfresh = relevant_features_tsfresh.values

        os.makedirs(os.path.dirname(x_tsfresh_path), exist_ok=True)
        os.makedirs(os.path.dirname(y_tsfresh_path), exist_ok=True)

        with open(x_tsfresh_path, "wb") as f:
            pickle.dump(x_tsfresh, f)
        with open(y_tsfresh_path, "wb") as f:
            pickle.dump(y_tsfresh, f)

        print("Data saved successfully.")


    # model = logistic_regression()
    # model = random_forest_model()
    # model = svm_model()
    # model = mlp_model()

    custom_cross_validation(x_tsfresh, y_tsfresh_cat, n_test_samples=2, model=model, pca_variance=0.80)
    # custom_cross_validation_xgboost(x_tsfresh, y_tsfresh_cat, pca_variance=0.8)



if __name__ == '__main__':
    main()
