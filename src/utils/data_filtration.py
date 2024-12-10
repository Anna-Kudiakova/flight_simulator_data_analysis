import os
import pickle
import pandas as pd

from src.utils.config import TRIAL_CONDITION_DIR, PILOT_METADATA_DIR


def load_trial_conditions():
    trial_conditions = pd.read_csv(TRIAL_CONDITION_DIR, delimiter=',')
    return trial_conditions


def filter_trial_conditions(trial_conditions, difficulty='NO WIND', outcome='SUCCESSFUL'):
    filtered_trials = trial_conditions[
        (trial_conditions['Difficulty'] == difficulty) &
        (trial_conditions['Outcome'] == outcome)
        ]
    return filtered_trials


def get_filtered_participant_data(participants_data, filtered_trials):
    filtered_participants_data = {}
    missing_participants = set()

    print('Trial conditions file - START')
    for _, row in filtered_trials.iterrows():
        participant = row['Subject']
        trial_num = row['Trial']

        if participant in participants_data and participant not in missing_participants:
            trial_key = f"{participant}_{trial_num}"
            if trial_key in participants_data[participant]:
                if participant not in filtered_participants_data:
                    filtered_participants_data[participant] = {}
                filtered_participants_data[participant][trial_key] = participants_data[participant][trial_key]
            # else:
            #     print(f"Trial '{trial_key}' not found for participant '{participant}'.")
        else:
            if participant not in missing_participants:
                # print(f"Participant '{participant}' not found in participants_data.")
                missing_participants.add(participant)

    print('Trial conditions file - END')

    return filtered_participants_data


def load_pilot_metadata():
    pilot_metadata = pd.read_csv(PILOT_METADATA_DIR, delimiter=',')
    pilot_metadata.dropna(how='all', inplace=True)
    return pilot_metadata


def categorize_pilots(pilot_metadata):
    novice_pilots = pilot_metadata[pd.isna(pilot_metadata['License'])]['Subject'].tolist()
    experienced_pilots = pilot_metadata[pd.notna(pilot_metadata['License'])]['Subject'].tolist()
    return novice_pilots, experienced_pilots

