import os
import pandas as pd


def process_csv_files(folder_path):
    combined_data = pd.DataFrame()

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            fingering = filename.split("_")[-1].split(".")[0]
            df = pd.read_csv(
                file_path,
                skiprows=1,
                header=None,
                names=["Frame", "Hand", "Landmark", "X", "Y", "Z", "Fingering"],
            )
            frame_counts = df.groupby("Frame").size()

            # Filter out frames with exactly 42 rows (2 hands * 21 landmarks)
            valid_frames = frame_counts[frame_counts == 42].index

            df = df[df["Frame"].isin(valid_frames)]
            df["Fingering"] = fingering
            combined_data = pd.concat([combined_data, df], ignore_index=True)

    return combined_data


# Path to the folder containing CSV files
folder_path = "landmark_data_stable"

combined_data = process_csv_files(folder_path)
combined_data.to_csv("combined_data_fitlered.csv", index=False)
print("Combined data saved to 'combined_data_filtered.csv'")
