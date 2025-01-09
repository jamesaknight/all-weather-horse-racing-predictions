import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =========================================================================
# 1) LOAD YOUR DATA
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
# 5) BUILD A SIMPLE NEURAL NETWORK WITH PYTORCH
# =========================================================================
class RacePredictionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RacePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Model initialization
input_dim = X_train_final.shape[1]
output_dim = 20  # Predicting 20 horses' LOG DTW+ values
model = RacePredictionModel(input_dim, output_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================================================================
# 6) TRAIN THE MODEL
# =========================================================================
# Convert data to torch tensors
X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_final, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Print average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}")

# =========================================================================
# 7) EVALUATE ON THE TEST SET
# =========================================================================
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)

print(f"Test MSE (loss): {test_loss.item():.4f}")

# =========================================================================
# 8) PREDICT FOR RACES WITH MISSING LOG DTW+
# =========================================================================
X_pred_tensor = torch.tensor(df_pred[X_cols].values.astype(float), dtype=torch.float32)
model.eval()
with torch.no_grad():
    preds = model(X_pred_tensor)

# Keep only Race_Level_Race Time, Horse_1_Horse to Horse_20_Horse, and predictions in the output
df_output = df_pred[["Race_Level_Race Time"]].copy()

for i in range(1, 21):
    horse_col = f"Horse_{i}_Horse"
    if horse_col in df_pred.columns:
        df_output[horse_col] = df_pred[horse_col]
    df_output[f"Horse_{i}_LOG DTW+_PRED"] = preds[:, i - 1].numpy()

# Save predictions
df_output.to_csv("predictions/predictions_for_missing_LOGDTW.csv", index=False)
print("Done! Predictions saved in 'predictions/predictions_for_missing_LOGDTW.csv'.")
