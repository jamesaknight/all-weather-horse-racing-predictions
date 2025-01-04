import pandas as pd

# Load the CSV file
input_file = '../data/2025_01_02.csv'
output_file = '../data/transformed_data.csv'

# Read the CSV into a DataFrame
data = pd.read_csv(input_file, header=None)

# The first row contains the column names
column_names = data.iloc[0]

# The second row (index 1) contains the operation for each column
operations = data.iloc[1]

# The third row (index 2) contains additional metadata like 'Race level', 'Target', 'Horse level'
metadata = data.iloc[2]

data = data.drop(index=[0, 1, 2]).reset_index(drop=True)  # Drop the first three rows

transformed_data = pd.DataFrame()
metadata_row = []

for col_index, operation in enumerate(operations):
    column_data = data[col_index]
    original_name = column_names[col_index]
    metadata_value = metadata[col_index]

    if operation.lower() == 'keep':
        # Retain the column with its original name
        transformed_data[original_name] = column_data
        metadata_row.append(metadata_value)

    elif operation.lower() == 'one hot':
        # One-hot encode the column with 0/1 values
        one_hot = pd.get_dummies(column_data, prefix=original_name, dtype=int)
        transformed_data = pd.concat([transformed_data, one_hot], axis=1)
        # Add metadata for each new one-hot column
        for unique_value in one_hot.columns:
            metadata_row.append(metadata_value)

    elif operation.lower() == 'discard':
        # Skip this column
        continue

    elif operation.lower() == 'encode':
        # Encode unique values to numbers
        mapping = {value: idx for idx, value in enumerate(column_data.unique())}
        transformed_data[original_name] = column_data.map(mapping)
        metadata_row.append(metadata_value)

# Add the metadata row back to the transformed data
transformed_data.loc[-1] = metadata_row  # Insert metadata row
transformed_data.index = transformed_data.index + 1  # Shift index
transformed_data.sort_index(inplace=True)  # Reorder to keep metadata as the second row

# Save the transformed data to a new CSV file
transformed_data.to_csv(output_file, index=False)

print(f"Data transformation complete. Transformed file saved to {output_file}")
