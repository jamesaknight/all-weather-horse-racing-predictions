import pandas as pd

# Load the transformed CSV file
input_file = '../data/AW_Processed_Normalized.csv'
output_file = '../data/final_race_data.csv'

# Read the CSV into a DataFrame without headers (use low_memory=False to handle mixed types)
data = pd.read_csv(input_file, header=None, low_memory=False)

# The first row contains the actual column names, and the second row contains metadata
data.columns = data.iloc[0]  # Set the first row as column names
metadata_row = data.iloc[1]  # The second row contains metadata

# Remove the first two rows (header and metadata row)
data = data.iloc[2:].reset_index(drop=True)

# Identify column types based on metadata
race_level_cols = metadata_row[metadata_row == 'Race Level'].index
horse_level_cols = metadata_row[metadata_row == 'Horse Level'].index
target_cols = metadata_row[metadata_row == 'Target Variable'].index

# Constants
max_horses = 20
num_race_level_features = len(race_level_cols)
num_horse_level_features = len(horse_level_cols)
num_target_features = len(target_cols)
expected_columns = (
        num_race_level_features +
        max_horses * num_horse_level_features +
        max_horses * num_target_features
)

# Group data by unique combinations of "Race Time" and course columns
race_time_col = 'Race Time'
course_cols = [col for col in data.columns if col.startswith('Course_')]

if race_time_col not in data.columns:
    raise ValueError("The dataset must contain a 'Race Time' column for unique race identification.")

unique_races = data.groupby([race_time_col] + course_cols)

final_data = []
headers = []

for group_key, group in unique_races:
    # Initialize the row for this unique race
    race_row = []

    # Add Race-Level Features
    race_level_data = group.iloc[0][race_level_cols].tolist()
    race_row.extend(race_level_data)

    # Add Horse-Level Features for all runners
    horse_level_data = group[horse_level_cols].values.flatten().tolist()

    # Pad Horse-Level Data
    pad_length = max_horses * num_horse_level_features - len(horse_level_data)
    horse_level_data.extend([-1] * pad_length)
    race_row.extend(horse_level_data)

    # Add Target Variable Features for all runners
    target_data = group[target_cols].values.flatten().tolist()

    # Pad Target Data
    pad_length = max_horses * num_target_features - len(target_data)
    target_data.extend([-1] * pad_length)
    race_row.extend(target_data)

    # Append the row to the dataset
    final_data.append(race_row)

    # Generate Headers Once
    if not headers:
        headers = (
                [f"Race_Level_{col}" for col in race_level_cols] +
                [f"Horse_{i+1}_{col}" for i in range(max_horses) for col in horse_level_cols] +
                [f"Horse_{i+1}_{col}" for i in range(max_horses) for col in target_cols]
        )

# Validate Output Dimensions
print(f"Number of Race Level Features: {num_race_level_features}")
print(f"Number of Horse Level Features (per horse): {num_horse_level_features}")
print(f"Number of Target Features (per horse): {num_target_features}")
print(f"Expected Columns: {expected_columns}")
print(f"Generated Columns: {len(headers)}")
if len(headers) != expected_columns:
    raise ValueError("Header column count mismatch.")

# Convert to DataFrame
final_df = pd.DataFrame(final_data, columns=headers)

# Save to CSV
final_df.to_csv(output_file, index=False)

print(f"Final race data transformation complete. File saved to {output_file}.")