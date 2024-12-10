from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features

from src.machine_learning.machine_learning_functions import prepare_tsfresh_data, encode_labels, \
    select_relevant_features, define_pca_components_number, hyperparameter_tuning
from src.utils.config import PROCESSED_DATA_DIR_EXTENDED, COLUMNS_DIR_EXTENDED
from src.utils.data_filtration import load_pilot_metadata
from src.utils.data_loader import load_or_process_data


def main():
    data_dir = PROCESSED_DATA_DIR_EXTENDED
    columns_dir = COLUMNS_DIR_EXTENDED
    participants_data = load_or_process_data(data_dir, columns_dir)

    df_tsfresh = prepare_tsfresh_data(participants_data)

    df_melted = df_tsfresh.melt(id_vars=['Time', 'pilot', 'trial'],
                                var_name='variable',
                                value_name='value')

    print("Extracting features with tsfresh...")
    extracted_features = extract_features(df_melted, column_id='pilot', column_sort='Time',
                                          column_kind='variable', column_value='value')
    extracted_features.fillna(0, inplace=True)  # Simple imputation

    print(f"Extracted features shape: {extracted_features.shape}")
    pilot_metadata = load_pilot_metadata()
    # Prepare labels for tsfresh (flight hours per pilot)
    y_tsfresh = pilot_metadata.set_index('Subject').loc[extracted_features.index, 'Hours']

    y_tsfresh_cat = encode_labels(y_tsfresh)

    relevant_features_tsfresh = select_relevant_features(extracted_features, y_tsfresh_cat)

    x_tsfresh = relevant_features_tsfresh.values
    scaler_tsfresh = StandardScaler()
    x_tsfresh_scaled = scaler_tsfresh.fit_transform(x_tsfresh)
    n_components_98 = define_pca_components_number(x_tsfresh_scaled, 0.98)
    pca_tsfresh = PCA(n_components=n_components_98, random_state=42)
    x_tsfresh_pca = pca_tsfresh.fit_transform(x_tsfresh_scaled)
    best_clf_tsfresh = hyperparameter_tuning(x_tsfresh_pca, y_tsfresh_cat)


if __name__ == '__main__':
    main()
