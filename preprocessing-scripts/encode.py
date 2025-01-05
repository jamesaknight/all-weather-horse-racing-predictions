import pandas as pd

# Load the dataset
aw_data = pd.read_csv('AW_Master.csv')

# --------------------------------------------------------------------------------
# ONE-HOT ENCODING 'Course' (cast booleans to 1/0)
# --------------------------------------------------------------------------------
encoded_courses = pd.get_dummies(aw_data['Course'], prefix='Course').astype(int)
aw_data = pd.concat([aw_data, encoded_courses], axis=1)
aw_data = aw_data.drop(columns=['Course'])

# --------------------------------------------------------------------------------
# RACE TYPE ENCODING
# --------------------------------------------------------------------------------
race_type_mapping = {
    'A/W': 0,
    'Turf': 1,
    'Jumps': 2,
    'NH Flat': 2
}

aw_data['Race Type'] = aw_data['Race Type'].map(race_type_mapping)
aw_data = aw_data[aw_data['Race Type'] == 0]  # Keep only A/W races

aw_data['Race Type LTO'] = aw_data['Race Type LTO'].map(race_type_mapping).fillna(-1).astype(int)

# --------------------------------------------------------------------------------
# CLASS ENCODING
# --------------------------------------------------------------------------------
class_mapping = {letter: idx + 1 for idx, letter in enumerate("ABCDEFG")}
aw_data['Class'] = aw_data['Class'].map(class_mapping)
aw_data['Class LTO (Same Code)'] = aw_data['Class LTO (Same Code)'].map(class_mapping).fillna(-1).astype(int)

# --------------------------------------------------------------------------------
# GOING ENCODING
# --------------------------------------------------------------------------------
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

aw_data['Going'] = aw_data['Going'].map(going_mapping).fillna(-1).astype(int)
aw_data['Actual Going LTO'] = aw_data['Actual Going LTO'].map(going_mapping).fillna(-1).astype(int)

# --------------------------------------------------------------------------------
# CREATE 'Today Pace IV' (based on 'Pace Rating')
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# CREATE 'LTO Pace IV' (based on 'Pace String')
# --------------------------------------------------------------------------------
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
aw_data = aw_data.drop(columns=['LTO Pace IV Leaders', 'LTO Pace IV Prominent',
                                'LTO Pace IV Held Up', 'Pace String'])

# Drop 'Jockey' and 'Trainer' columns
aw_data = aw_data.drop(columns=['Jockey', 'Trainer'])

# --------------------------------------------------------------------------------
# CREATE 'PRB' COLUMN
# --------------------------------------------------------------------------------
def calculate_prb(row):
    position = row['Position']
    runners = row['Runners']
    if position > 50:  # Horse failed to finish
        return 0.0
    return round((position - 1) / (runners - 1), 2)

aw_data['PRB'] = aw_data.apply(calculate_prb, axis=1)

# --------------------------------------------------------------------------------
# HANDLE MISSING 'DSLR'
# --------------------------------------------------------------------------------
aw_data['DSLR'] = aw_data['DSLR'].fillna(-1).astype(int)

# --------------------------------------------------------------------------------
# CREATE 'Equip Change' COLUMN
# --------------------------------------------------------------------------------
def calculate_equip_change(row):
    if '1' in str(row['Equip']):  # Check if 'Equip' contains '1'
        return 2
    equip_letters = ''.join([char for char in str(row['Equip']) if char.isalpha()])
    headgear_letters = ''.join([char for char in str(row['Headgear LTO']) if char.isalpha()])

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

# --------------------------------------------------------------------------------
# NORMALIZE 'PRB' COLUMNS
# --------------------------------------------------------------------------------
prb_columns = [
    'All PRB', "Today's Going PRB", "Today's Distance PRB",
    "Today's Class PRB", "Today's Course PRB"
]
for col in prb_columns:
    aw_data[col] = aw_data[col].div(100).round(2).fillna(-1).astype(float)

# --------------------------------------------------------------------------------
# COMPUTE BFSP% AND BFSP% LTO
# --------------------------------------------------------------------------------
aw_data['BFSP%'] = aw_data['BF Decimal SP'].apply(lambda x: round(1 / x, 3) if pd.notna(x) and x > 0 else -1)
aw_data['BFSP% LTO'] = aw_data['Betfair SP Odds LTO (Numerical) (Same Code)'].apply(
    lambda x: round(1 / x, 3) if pd.notna(x) and x > 0 else -1
)

# Drop the original BF columns
aw_data = aw_data.drop(columns=['BF Decimal SP', 'Betfair SP Odds LTO (Numerical) (Same Code)'])

# --------------------------------------------------------------------------------
# ENCODE 'Sex Abbrev'
# --------------------------------------------------------------------------------
aw_data['Sex Abbrev'] = aw_data['Sex Abbrev'].apply(lambda x: 1 if str(x).lower() in ['f', 'm'] else 0)

# --------------------------------------------------------------------------------
# ENCODE 'HCP' AND 'HCP LTO'
# --------------------------------------------------------------------------------
aw_data['HCP'] = aw_data['Handicap or Non Handicap'].apply(
    lambda x: 1 if str(x).strip().upper() == 'HANDICAP' else
    (0 if str(x).strip().upper() == 'NON HANDICAP' else -1)
)
aw_data['HCP LTO'] = aw_data['Handicap or Non Handicap Last Time Out'].apply(
    lambda x: 1 if str(x).strip().upper() == 'HANDICAP' else
    (0 if str(x).strip().upper() == 'NON HANDICAP' else -1)
)

# --------------------------------------------------------------------------------
# CREATE 'Distance v LTO'
# --------------------------------------------------------------------------------
aw_data['Distance v LTO'] = aw_data['Distance (y)'] - aw_data['Distance In Yards LTO (Same Code)']

# Drop 'Distance In Yards LTO (Same Code)'
aw_data = aw_data.drop(columns=['Distance In Yards LTO (Same Code)'])

# --------------------------------------------------------------------------------
# CREATE 'DTW LTO Hcp' COLUMN
# --------------------------------------------------------------------------------
aw_data['DTW LTO Hcp'] = aw_data.apply(
    lambda row: row['Distance To Winner LTO (Same Code)'] if row['HCP LTO'] == 1 else -1, axis=1
)

# --------------------------------------------------------------------------------
# Create 'Market LTO Hcp' column
aw_data['Market LTO Hcp'] = (
    aw_data.apply(
        lambda row: row['BFSP% LTO'] * row['Number of Runners LTO (Same Code)']
        if row['HCP LTO'] == 1 else -1,
        axis=1
    )
    .astype(float)  # Ensure numeric float type
    .round(2)       # Round to 2 decimal places
)

# --------------------------------------------------------------------------------
# ROUND CERTAIN COLUMNS TO 3 DECIMAL PLACES
# --------------------------------------------------------------------------------
columns_to_format = ['Draw IV', 'DTW', 'Distance To Winner LTO (Same Code)', 'DTW LTO Hcp']
for col in columns_to_format:
    if col in aw_data.columns:
        aw_data[col] = aw_data[col].round(3)

# --------------------------------------------------------------------------------
# IF 'Runs Before' = 0, SET SPECIFIED COLUMNS TO -1
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# DROP ORIGINAL 'Handicap' COLUMNS AND OTHERS
# --------------------------------------------------------------------------------
aw_data = aw_data.drop(columns=[
    'Handicap or Non Handicap',
    'Handicap or Non Handicap Last Time Out',
    'History (CD BF etc)',
    'OR',
    'Distance To Winner LTO'
])

# --------------------------------------------------------------------------------
# Create the 'PPI' (Pace Pressure Index) column
# Group by Race Time, sum the Pace Rating, then divide by Runners, and round to 2 decimals
aw_data['PPI'] = (
        aw_data.groupby('Race Time')['Pace Rating']
        .transform('sum') / aw_data['Runners']
).round(2)

# --------------------------------------------------------------------------------
# SAVE THE UPDATED DATASET
# --------------------------------------------------------------------------------
aw_data.to_csv('AW_Processed.csv', index=False)
print("Processed data saved to: AW_Processed.csv")