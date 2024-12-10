# src/utils/data_loader.py

import os
import pickle

import pandas as pd

from src.utils.config import CSV_DELIMITER, COLUMNS_DIR, DATA_DIR, PROCESSED_DATA_DIR


def get_headers(file_path):
    df = pd.read_csv(file_path, delimiter=";")
    columns = df.columns.tolist()
    return columns


def find_csv_files(data_dir):
    csv_files = []

    # Traverse the participant folders
    for participant in os.listdir(data_dir):
        participant_folder = os.path.join(data_dir, participant)

        if os.path.isdir(participant_folder):
            for trial in os.listdir(participant_folder):
                trial_folder = os.path.join(participant_folder, trial)

                if os.path.isdir(trial_folder):
                    for file in os.listdir(trial_folder):
                        if file.endswith(".csv"):
                            csv_files.append({
                                "participant": participant,
                                "trial": trial,
                                "file_path": os.path.join(trial_folder, file)
                            })
    return csv_files


def load_csv_files(csv_info, columns_dir):
    columns = get_headers(columns_dir)
    participants_data = {}

    for csv_meta in csv_info:
        participant = csv_meta['participant']
        trial = csv_meta['trial']
        csv_path = csv_meta['file_path']
        trial_name = f"{participant}_{trial.split('_')[-1]}"

        try:
            df = pd.read_csv(csv_path, delimiter=CSV_DELIMITER, usecols=lambda col: col in columns)
            print(f'Trial # {trial_name} - {df.shape[1]}')
            if participant not in participants_data:
                participants_data[participant] = {}

            participants_data[participant][trial_name] = df

        except Exception as e:

            print(f"Error while processing trial: {trial_name}")
            print(f"Exception: {e}")

    return participants_data


def load_all_data(columns_dir):
    csv_files = find_csv_files(DATA_DIR)
    data_frames = load_csv_files(csv_files, columns_dir)
    return data_frames


def save_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {file_path}")


def load_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    print(f"Data loaded from {file_path}")
    return data


def load_or_process_data(data_dir, columns_dir):
    if os.path.exists(data_dir):
        print("Pickle file found. Loading data from the pickle file...")
        return load_data(data_dir)
    else:
        print("Pickle file not found. Processing data from scratch...")
        participants_data = load_all_data(columns_dir)
        save_data(participants_data, data_dir)
        return participants_data



