import pandas as pd

# Load the transformed CSV file
input_file = '../data/transformed_data.csv'
output_file = '../data/final_race_data.csv'

# Read the CSV into a DataFrame without headers (since we will set the header manually)
data = pd.read_csv(input_file, header=None)

# The first row contains the actual column names, and the second row contains metadata
data.columns = data.iloc[0]  # Set the first row as column names
metadata_row = data.iloc[1]  # The second row contains metadata

# Remove the first two rows (header and metadata row)
data = data.iloc[2:].reset_index(drop=True)

# Identify column types based on metadata
race_level_cols = metadata_row[metadata_row == 'Race level'].index
horse_level_cols = metadata_row[metadata_row == 'Horse level'].index
target_cols = metadata_row[(metadata_row == 'Target') | (metadata_row == 'Target?')].index

# Identify "Race time" column and one-hot encoded course columns
race_time_col = 'Race Time'
course_cols = data.columns[2:8]  # Assuming columns 2-7 are one-hot encoded courses

if race_time_col not in data.columns:
    raise ValueError("The dataset must contain a 'Race Time' column for unique race identification.")

# Group data by unique combinations of "Race time" and courses
unique_races = data.groupby([race_time_col] + course_cols.tolist())

final_data = []
first_race = True  # Flag to ensure headers are added only for the first race

# Set the maximum number of horses for each race (20)
max_horses = 20

for group_key, group in unique_races:
    # Initialize the row for this unique race
    race_row = []

    # Add "Race level" data from the first row of the group
    race_level_data = group.iloc[0][race_level_cols].tolist()
    race_row.extend(race_level_data)

    # Add "Horse level" data for all rows in the group
    horse_level_data = group[horse_level_cols].values.flatten().tolist()

    # Pad horse_level_data if there are fewer than max_horses
    if len(horse_level_data) < max_horses * len(horse_level_cols):
        # Calculate how many entries to pad
        pad_length = max_horses * len(horse_level_cols) - len(horse_level_data)
        horse_level_data.extend([-1] * pad_length)

    # Add the horse-level data to the race row
    race_row.extend(horse_level_data)

    # Add "Target" and "Target?" data in order
    target_data = group.iloc[0][target_cols].tolist()
    race_row.extend(target_data)

    # Append the completed row to the final dataset
    final_data.append(race_row)

    # Only set headers for the first race
    if first_race:
        headers = (
                metadata_row[race_level_cols].tolist() +
                [f"Horse_{i+1}_{col}" for i in range(max_horses) for col in horse_level_cols] +
                metadata_row[target_cols].tolist()
        )
        first_race = False  # After the first race, headers will not be added again

# Check if headers and final_data column count match
if len(headers) != len(final_data[0]):
    raise ValueError("Mismatch between number of columns in final_data and headers.")

# Create the final DataFrame
final_df = pd.DataFrame(final_data, columns=headers)

# Save the final data to a new CSV file
final_df.to_csv(output_file, index=False)

print(f"Final race data transformation complete. File saved to {output_file}")
