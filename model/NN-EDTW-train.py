import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# =========================================================================
# 1) LOAD YOUR DATA
#    - We assume it has 1 row per race, with columns:
#      - Race-Level: e.g. Race_Level_Distance (y), Race_Level_Class, Race_Level_HCP, etc.
#      - Horse-Level: e.g. Horse_1_Draw IV, Horse_1_Trn Stats, ...
#      - Target columns: Horse_1_LOG DTW+, ..., Horse_20_LOG DTW+.
# =========================================================================

df = pd.read_csv("../data/final_race_data.csv")

# =========================================================================
# NEW STEP: KEEP ONLY RACES WHERE RACE_Level_HCP = 1
# =========================================================================
df = df[df["Race_Level_HCP"] == 1].copy()

# =========================================================================
# 2) DEFINE THE RACE-LEVEL AND HORSE-LEVEL FEATURES
# =========================================================================

race_level_features = [
    "Race_Level_Distance (y)",
    "Race_Level_Class",
    "Race_Level_Course_Chelmsford City",
    "Race_Level_Course_Kempton",
    "Race_Level_Course_Lingfield",
    "Race_Level_Course_Newcastle",
    "Race_Level_Course_Southwell",
    "Race_Level_Course_Wolverhampton",
]

base_horse_level_features = [
    "Draw IV",
    "Trn Stats",
    "Jky Stats",
    "MR Last 3 Runs Speed Rating",
    "LTO Speed Rating",
    "Today's Course PRB",
]

# Construct the full list of input columns (X)
X_cols = race_level_features.copy()
for i in range(1, 21):
    for feat in base_horse_level_features:
        col_name = f"Horse_{i}_{feat}"
        X_cols.append(col_name)

# Construct the target columns (Y)
y_cols = [f"Horse_{i}_LOG DTW+" for i in range(1, 21)]

# =========================================================================
# 3) FILTER / PREPARE DATA
# =========================================================================
df_train = df.dropna(subset=y_cols, how='any').copy()
df_pred = df[df[y_cols].isnull().any(axis=1)].copy()

# =========================================================================
# 4) SPLIT INTO X, Y ARRAYS FOR TRAINING/VALIDATION/TEST
#    - We will create three sets: train, validation, and test.
# =========================================================================

X_full = df_train[X_cols].values.astype(float)
y_full = df_train[y_cols].values.astype(float)

# First, split off a test set (20% of total)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

# Next, split the remainder (80%) into train (80% of 80%=64%) and val (20% of 80%=16%)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

print("Shapes:")
print("  X_train_final:", X_train_final.shape)
print("  y_train_final:", y_train_final.shape)
print("  X_val:", X_val.shape)
print("  y_val:", y_val.shape)
print("  X_test:", X_test.shape)
print("  y_test:", y_test.shape)

# =========================================================================
# 5) BUILD A SIMPLE NEURAL NETWORK
# =========================================================================
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_final.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(20, activation='linear'))

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

model.summary()

# =========================================================================
# 6) TRAIN THE MODEL
# =========================================================================
history = model.fit(
    X_train_final,
    y_train_final,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# =========================================================================
# 7) EVALUATE ON THE TEST SET
# =========================================================================
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE (loss): {test_loss:.4f}  |  Test MAE: {test_mae:.4f}")

# =========================================================================
# 8) PREDICT FOR RACES WITH MISSING LOG DTW+
# =========================================================================
X_pred = df_pred[X_cols].values.astype(float)
preds = model.predict(X_pred)

# Keep only Race_Level_Race Time, Horse_1_Horse to Horse_20_Horse, and predictions in the output
df_output = df_pred[["Race_Level_Race Time"]].copy()

for i in range(1, 21):
    horse_col = f"Horse_{i}_Horse"
    if horse_col in df_pred.columns:
        df_output[horse_col] = df_pred[horse_col]
    df_output[f"Horse_{i}_LOG DTW+_PRED"] = preds[:, i - 1]

# Save predictions
df_output.to_csv("predictions_for_missing_LOGDTW.csv", index=False)
print("Done! Predictions saved in 'predictions_for_missing_LOGDTW.csv'.")
