from src.utils.config import PROCESSED_DATA_DIR_EXTENDED, COLUMNS_DIR_EXTENDED
from src.utils.data_filtration import load_pilot_metadata
from src.utils.data_loader import load_or_process_data


def main():
    data_dir = PROCESSED_DATA_DIR_EXTENDED
    columns_dir = COLUMNS_DIR_EXTENDED
    participants_data = load_or_process_data(data_dir, columns_dir)

    print(f"Participant number: {len(participants_data)}")
    trials_num = 0

    for participant, trials in participants_data.items():
        print(f"Participant: {participant}")
        print(f"Number of trials: {len(trials)}")
        trials_num += len(trials)
    print(f"Total number of trials: {trials_num}")
    print(f"Average number of trials per participant: {trials_num / len(participants_data)}")

    pilot_metadata = load_pilot_metadata()
    curr_pilot_metadata = pilot_metadata[pilot_metadata["Subject"].isin(participants_data.keys())]

    novice = 0
    private = 0
    intermediate = 0
    commercial = 0

    novice_pilots = {}
    private_pilots = {}
    intermediate_pilots = {}
    commercial_pilots = {}

    for index, row in curr_pilot_metadata.iterrows():
        subject = row["Subject"]
        hours = int(row["Hours"])

        # Classify the pilot based on hours
        if hours < 40:
            novice_pilots[subject] = hours
        elif 40 <= hours < 140:
            private_pilots[subject] = hours
        elif 140 <= hours < 240:
            intermediate_pilots[subject] = hours
        else:  # hours >= 240
            commercial_pilots[subject] = hours

    # Print summary
    print("Novice Pilots (0-40 hours):")
    for pilot, hrs in novice_pilots.items():
        print(f"{pilot}: {hrs} hours")

    print("\nPrivate Pilots (40-140 hours):")
    for pilot, hrs in private_pilots.items():
        print(f"{pilot}: {hrs} hours")

    print("\nIntermediate Pilots (140-240 hours):")
    for pilot, hrs in intermediate_pilots.items():
        print(f"{pilot}: {hrs} hours")

    print("\nCommercial Pilots (> 240 hours):")
    for pilot, hrs in commercial_pilots.items():
        print(f"{pilot}: {hrs} hours")





    # x_stat, y_stat = extract_statistical_features(participants_data)


if __name__ == "__main__":
    main()
