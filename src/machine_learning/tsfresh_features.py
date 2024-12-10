import os
import pickle

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

from src.machine_learning.machine_learning_functions import prepare_tsfresh_data, encode_labels, \
    select_relevant_features, define_pca_components_number, hyperparameter_tuning
from src.utils.config import PROCESSED_DATA_DIR_EXTENDED, COLUMNS_DIR_EXTENDED, TSFRESH_DIR, X_TSFRESH_DIR, \
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

        with open(x_tsfresh_path, "wb") as f:
            pickle.dump(x_tsfresh, f)
        with open(y_tsfresh_path, "wb") as f:
            pickle.dump(y_tsfresh, f)

        print("Data saved successfully.")

    scaler_tsfresh = StandardScaler()
    x_tsfresh_scaled = scaler_tsfresh.fit_transform(x_tsfresh)
    n_components_98 = define_pca_components_number(x_tsfresh_scaled, 0.98)
    pca_tsfresh = PCA(n_components=n_components_98, random_state=42)
    x_tsfresh_pca = pca_tsfresh.fit_transform(x_tsfresh_scaled)
    best_clf_tsfresh = hyperparameter_tuning(x_tsfresh_pca, y_tsfresh_cat)

    y_pred = best_clf_tsfresh.predict(x_tsfresh_pca)

    # Confusion matrix
    cm = confusion_matrix(y_tsfresh_cat, y_pred, labels=y_tsfresh_cat.cat.categories)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_tsfresh_cat.cat.categories)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    main()
