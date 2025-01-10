import pandas as pd
import numpy as np

# Define the base horse-level features
base_horse_level_features = [
    "Trn Stats",
    "Jky Stats",
    "MR Last 3 Runs Speed Rating",
    "LTO Speed Rating",
    "Today's Course PRB",
    "PRC Average",
    "PRC Last Run",
    "PRC 2nd Last Run",
    "PRC 3rd Last Run",
    "Pace Rating",
    "Weight (pounds)",
    "Age",
    "DSLR",
    "Draw",
    "Draw IV",
]


# =========================================================================
# FUNCTION TO REARRANGE HORSES BY DRAW
# =========================================================================
def rearrange_horses_by_draw(df, num_horses=20):
    rearranged_races = []

    for _, race in df.groupby("Race_Level_Race Time"):  # Group by race
        print(race)
        race_horse_data = []

        # Preserve race-level features
        race_level_features = {
            col: race[col].iloc[0] for col in race.columns if col.startswith("Race_Level_")
        }

        for i in range(1, num_horses + 1):
            draw_col = f"Horse_{i}_Draw"

            # If the draw column is present, clean its values
            if draw_col in race.columns:
                draw_value = race[draw_col].values[0] if not race[draw_col].empty else np.inf
                draw_value = np.inf if draw_value == -1 else draw_value
            else:
                draw_value = np.inf  # Default to infinity if column is missing

            # Gather all features for this horse (include targets)
            horse_features = {
                f"Horse_{i}_{feat}": race[f"Horse_{i}_{feat}"].values[0]
                if f"Horse_{i}_{feat}" in race.columns else np.nan
                for feat in base_horse_level_features + [f"LOG DTW+"]
            }
            horse_features["Draw"] = draw_value
            race_horse_data.append(horse_features)

        # Sort horses by their draw values (smallest first, `np.inf` last)
        sorted_horses = sorted(race_horse_data, key=lambda x: x["Draw"])

        # Flatten the sorted data into a single dictionary (row format)
        sorted_row = {}
        for i, horse in enumerate(sorted_horses, start=1):
            for key, value in horse.items():
                if key == "Draw" and value == np.inf:
                    value = -1  # Restore -1 for infinity draw values
                sorted_row[key.replace(f"Horse_{i}_", f"Horse_{i}_")] = value

        # Add race-level features to the row
        sorted_row.update(race_level_features)
        rearranged_races.append(sorted_row)

    return pd.DataFrame(rearranged_races)


# =========================================================================
# MAIN FUNCTION FOR PROCESSING DATA
# =========================================================================
if __name__ == "__main__":
    input_file = "../data/final_race_data.csv"
    output_train_file = "../data/processed_train_data.csv"
    output_pred_file = "../data/processed_pred_data.csv"

    # Load the data
    df = pd.read_csv(input_file)

    # Filter races with Race_Level_HCP = 1
    df = df[df["Race_Level_HCP"] == 1].copy()

    # Define the target columns (LOG DTW+ values)
    y_cols = [f"Horse_{i}_LOG DTW+" for i in range(1, 21)]

    # Separate train and prediction datasets
    df_train = df.dropna(subset=y_cols, how="any").copy()
    df_pred = df[df[y_cols].isnull().any(axis=1)].copy()

    # Rearrange horses by draw
    df_train_sorted = rearrange_horses_by_draw(df_train)
    df_pred_sorted = rearrange_horses_by_draw(df_pred)

    # Save the processed data
    df_train_sorted.to_csv(output_train_file, index=False)
    df_pred_sorted.to_csv(output_pred_file, index=False)

    print(f"Processed training data saved to {output_train_file}")
    print(f"Processed prediction data saved to {output_pred_file}")