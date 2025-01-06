import pandas as pd

# Load the dataset
aw_data = pd.read_csv('AW_Master.csv')

# One-hot encode the 'Course' column with 0/1 instead of True/False
encoded_courses = pd.get_dummies(aw_data['Course'], prefix='Course').astype(int)

# Add the one-hot encoded columns to the original DataFrame
aw_data = pd.concat([aw_data, encoded_courses], axis=1)

# Drop the original 'Course' column
aw_data = aw_data.drop(columns=['Course'])

# Define a mapping for 'Race Type' and 'Race Type LTO'
race_type_mapping = {
    'A/W': 0,
    'Turf': 1,
    'Jumps': 2,
    'NH Flat': 2
}

# Encode 'Race Type' and ensure all rows are included
aw_data['Race Type'] = aw_data['Race Type'].map(race_type_mapping)

# Filter the dataset to include only 'Race Type' = 0 (A/W races)
aw_data = aw_data[aw_data['Race Type'] == 0]

# Encode 'Race Type LTO', handling missing values as -1
aw_data['Race Type LTO'] = aw_data['Race Type LTO'].map(race_type_mapping).fillna(-1).astype(int)

# Define a mapping for 'Class' and 'Class LTO (Same Code)'
class_mapping = {letter: idx + 1 for idx, letter in enumerate("ABCDEFG")}

# Encode 'Class' column
aw_data['Class'] = aw_data['Class'].map(class_mapping)

# Encode 'Class LTO (Same Code)', handling missing values for unraced horses as -1
aw_data['Class LTO (Same Code)'] = aw_data['Class LTO (Same Code)'].map(class_mapping).fillna(-1).astype(int)

# Handle 'Draw' column: replace 0 with -1
aw_data['Draw'] = aw_data['Draw'].apply(lambda x: -1 if x == 0 else x)

# Handle 'Draw IV' column: replace blanks/missing values with -1
aw_data['Draw IV'] = aw_data['Draw IV'].fillna(-1).astype(float)

# Define a mapping for 'Going' and 'Actual Going LTO'
going_mapping = {
    'Heavy': 1,
    'Soft': 2, 'Slow': 2,
    'Good To Soft': 3,
    'Good': 4,
    'Standard': 4,
    'Good To Firm': 5,
    'Fast': 6,
    'Firm': 6
}

# Encode 'Going' column, handling missing values as 4 (Good/Standard)
aw_data['Going'] = aw_data['Going'].map(going_mapping).fillna(4).astype(int)

# Encode 'Actual Going LTO' column, handling missing values as 4 (Good/Standard)
aw_data['Actual Going LTO'] = aw_data['Actual Going LTO'].map(going_mapping).fillna(4).astype(int)


# Create the 'Today Pace IV' column based on the 'Pace Rating'
def calculate_today_pace_iv(row):
    if pd.isna(row['Pace Rating']):  # Handle missing 'Pace Rating'
        return -1
    elif row['Pace Rating'] in [0, 2]:
        return row['Pace IV Held Up']
    elif row['Pace Rating'] in [4, 6]:
        return row['Pace IV Prominent']
    elif row['Pace Rating'] >= 8:
        return row['Pace IV Led']
    else:
        return -1  # Default to -1 for unexpected values

aw_data['Today Pace IV'] = aw_data.apply(calculate_today_pace_iv, axis=1)

# Drop the original pace columns
aw_data = aw_data.drop(columns=['Pace IV Led', 'Pace IV Prominent', 'Pace IV Held Up'])

# Create the 'LTO Pace IV' column based on 'Pace String'
def calculate_lto_pace_iv(row):
    if pd.isna(row['Pace String']):  # Handle missing 'Pace String'
        return -1
    pace_string = row['Pace String'].replace('-', '').replace('/', '')  # Remove dashes and slashes
    if not pace_string:  # No letters in the string
        return -1
    last_letter = pace_string[-1]  # Get the far-right letter
    if last_letter == 'H':
        return row['LTO Pace IV Held Up']
    elif last_letter == 'P':
        return row['LTO Pace IV Prominent']
    elif last_letter == 'L':
        return row['LTO Pace IV Leaders']
    else:
        return -1  # Default to -1 for unexpected letters

aw_data['LTO Pace IV'] = aw_data.apply(calculate_lto_pace_iv, axis=1)

# Drop the original LTO pace columns and 'Pace String'
aw_data = aw_data.drop(columns=['LTO Pace IV Leaders', 'LTO Pace IV Prominent', 'LTO Pace IV Held Up', 'Pace String'])

# Drop 'Jockey' and 'Trainer' columns
aw_data = aw_data.drop(columns=['Jockey', 'Trainer'])

# Calculate the 'PRB' column
def calculate_prb(row):
    position = row['Position']
    runners = row['Runners']
    if position > 50:  # Horse failed to finish
        return 0.0
    return round((position - 1) / (runners - 1), 2)

aw_data['PRB'] = aw_data.apply(calculate_prb, axis=1)

# Cap 'DTW' values and handle failed finishers
aw_data['DTW'] = aw_data['DTW'].apply(lambda x: min(x, 30) if x > 0 else x)  # Cap DTW at 30
aw_data.loc[aw_data['Position'] > 50, 'DTW'] = 30  # Set DTW to 30 for failed finishers

# Cap 'Distance To Winner LTO (Same Code)' values and handle failed finishers
aw_data['Distance To Winner LTO (Same Code)'] = aw_data['Distance To Winner LTO (Same Code)'].apply(
    lambda x: min(x, 30) if pd.notna(x) else x  # Cap at 30
)
aw_data.loc[aw_data['FinPos LTO (Same Code)'] > 50, 'Distance To Winner LTO (Same Code)'] = 30  # Set to 30 for failed finishers

# Handle missing 'DSLR' entries
aw_data['DSLR'] = aw_data['DSLR'].fillna(-1).astype(int)

# Create 'Equip Change' column
def calculate_equip_change(row):
    if '1' in str(row['Equip']):  # Check if 'Equip' contains '1'
        return 2
    equip_letters = ''.join([char for char in str(row['Equip']) if char.isalpha()])  # Extract letters from 'Equip'
    headgear_letters = ''.join([char for char in str(row['Headgear LTO']) if char.isalpha()])  # Extract letters from 'Headgear LTO'

    # Check for one being blank and the other containing letters
    if (not equip_letters and headgear_letters) or (equip_letters and not headgear_letters):
        return 1

    # Check if letters differ
    if equip_letters != headgear_letters:
        return 1

    # Default to 0 if letters are the same or both are blank
    return 0

aw_data['Equip Change'] = aw_data.apply(calculate_equip_change, axis=1)

# Drop the original 'Equip' and 'Headgear LTO' columns
aw_data = aw_data.drop(columns=['Equip', 'Headgear LTO'])

# Normalize PRB columns and handle missing values
prb_columns = ['All PRB', 'Today\'s Going PRB', 'Today\'s Distance PRB', 
               'Today\'s Class PRB', 'Today\'s Course PRB']

for col in prb_columns:
    aw_data[col] = aw_data[col].div(100).round(2).fillna(-1).astype(float)

# Compute BFSP% and BFSP% LTO
aw_data['BFSP%'] = aw_data['BF Decimal SP'].apply(
    lambda x: round(1 / x, 3) if pd.notna(x) and x > 1 else (None if x == 1 else -1)
)

aw_data['BFSP% LTO'] = aw_data['Betfair SP Odds LTO (Numerical) (Same Code)'].apply(
    lambda x: round(1 / x, 3) if pd.notna(x) and x > 1 else (None if x == 1 else -1)
)

# Drop the original BF columns
aw_data = aw_data.drop(columns=['BF Decimal SP', 'Betfair SP Odds LTO (Numerical) (Same Code)'])


# Encode 'Sex Abbrev'
aw_data['Sex Abbrev'] = aw_data['Sex Abbrev'].apply(lambda x: 1 if str(x).lower() in ['f', 'm'] else 0)

# Encode 'HCP' and 'HCP LTO'
aw_data['HCP'] = aw_data['Handicap or Non Handicap'].apply(
    lambda x: 1 if str(x).strip().upper() == 'HANDICAP' else (0 if str(x).strip().upper() == 'NON HANDICAP' else -1)
)
aw_data['HCP LTO'] = aw_data['Handicap or Non Handicap Last Time Out'].apply(
    lambda x: 1 if str(x).strip().upper() == 'HANDICAP' else (0 if str(x).strip().upper() == 'NON HANDICAP' else -1)
)

# Create 'Distance v LTO'
aw_data['Distance v LTO'] = aw_data['Distance (y)'] - aw_data['Distance In Yards LTO (Same Code)']

# Drop 'Distance In Yards LTO (Same Code)'
aw_data = aw_data.drop(columns=['Distance In Yards LTO (Same Code)'])

# Create 'DTW LTO Hcp' column
aw_data['DTW LTO Hcp'] = aw_data.apply(
    lambda row: row['Distance To Winner LTO (Same Code)'] if row['HCP LTO'] == 1 else -1, axis=1
)

# Create 'Market LTO Hcp' column and ensure it is a float rounded to 3 decimal places
aw_data['Market LTO Hcp'] = aw_data.apply(
    lambda row: round(row['BFSP% LTO'] * row['Number of Runners LTO (Same Code)'], 3)
    if row['HCP LTO'] == 1 and pd.notna(row['BFSP% LTO']) and pd.notna(row['Number of Runners LTO (Same Code)'])
    else -1,
    axis=1
)


# Format specified columns to 3 decimal places
columns_to_format = ['Draw IV', 'DTW', 'Distance To Winner LTO (Same Code)', 'DTW LTO Hcp']
for col in columns_to_format:
    if col in aw_data.columns:
        aw_data[col] = aw_data[col].round(3)

# Handle 'Runs Before' = 0 and set specified columns to -1
columns_to_update = [
    'PRC Average', 'PRC Last Run', 'PRC 2nd Last Run', 'PRC 3rd Last Run',
    'HA Career Speed Rating', 'HA Career Speed Rating Rank', 
    'HA Last 1 Year Speed Rating', 'HA Last 1 Year Speed Rating Rank', 
    'MR Career Speed Rating', 'MR Career Speed Rating Rank', 
    'MR Last 1 Year Speed Rating', 'MR Last 1 Year Speed Rating Rank', 
    'MR Last 3 Runs Speed Rating', 'MR Last 3 Runs Speed Rating Rank', 
    'LTO Speed Rating', 'LTO Speed Rating Rank', '2nd LTO Speed Rating', 
    '2nd LTO Speed Rating Rank', '3rd LTO Speed Rating', 
    '3rd LTO Speed Rating Rank', '4th LTOt Speed Rating', 
    '4th LTO Speed Rating Rank'
]

aw_data.loc[aw_data['Runs Before'] == 0, columns_to_update] = -1

# Drop original 'Handicap' columns and additional ones
aw_data = aw_data.drop(columns=[
    'Handicap or Non Handicap', 
    'Handicap or Non Handicap Last Time Out', 
    'History (CD BF etc)', 
    'OR',
    'Distance To Winner LTO'
])

# Save the updated dataset to the same CSV file
aw_data.to_csv('AW_Processed.csv', index=False)

print("Processed data saved to: AW_Processed.csv")
