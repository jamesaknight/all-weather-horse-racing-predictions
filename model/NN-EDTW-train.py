import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error

# =========================================================================
# 1) LOAD PROCESSED DATA
# =========================================================================
train_file = "../data/processed_train_data.csv"
pred_file = "../data/processed_pred_data.csv"

df_train_sorted = pd.read_csv(train_file)
df_pred_sorted = pd.read_csv(pred_file)

# =========================================================================
# 2) DEFINE FEATURES AND TARGETS
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
    "Race_Level_PPI",
    "Race_Level_Going",
]

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

# Construct the full list of input columns (X)
X_cols = race_level_features.copy()
for i in range(1, 21):
    for feat in base_horse_level_features:
        col_name = f"Horse_{i}_{feat}"
        X_cols.append(col_name)

# Construct the target columns (Y)
y_cols = [f"Horse_{i}_LOG DTW+" for i in range(1, 21)]

# =========================================================================
# 3) SPLIT INTO X, Y ARRAYS FOR TRAINING/VALIDATION/TEST
# =========================================================================
X_full = df_train_sorted[X_cols].values.astype(float)
y_full = df_train_sorted[y_cols].values.astype(float)

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
# 4) BUILD A SIMPLE NEURAL NETWORK WITH PYTORCH
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

print(model)

# Loss and optimizer
criterion = nn.HuberLoss(delta=1.0)  # Huber Loss with delta=1.0
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# =========================================================================
# 5) TRAIN THE MODEL
# =========================================================================
# Convert data to torch tensors
X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_final, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Early stopping parameters
num_epochs = 400  # Maximum number of epochs
patience = 60  # Increased patience to allow for slightly longer training
delta = 1e-9  # Minimum improvement threshold for validation loss

best_val_loss = float('inf')
epochs_without_improvement = 0

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

    # Average loss for the epoch
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}")

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

    # Check for early stopping with a minimum improvement threshold
    if best_val_loss - val_loss > delta:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pth")  # Save the best model
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

# Load the best model
model.load_state_dict(torch.load("best_model.pth"))

# =========================================================================
# 6) EVALUATE ON THE TEST SET
# =========================================================================
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor).item()

# Convert predictions and ground truth back to NumPy arrays
test_outputs_np = test_outputs.numpy()
y_test_np = y_test_tensor.numpy()

# Compute additional metrics
r2 = r2_score(y_test_np, test_outputs_np)
mae = mean_absolute_error(y_test_np, test_outputs_np)

print(f"Test Loss (Huber): {test_loss:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test MAE: {mae:.4f}")

# =========================================================================
# 7) PREDICT FOR RACES WITH MISSING LOG DTW+
# =========================================================================
X_pred_tensor = torch.tensor(df_pred_sorted[X_cols].values.astype(float), dtype=torch.float32)
model.eval()
with torch.no_grad():
    preds = model(X_pred_tensor)

# Save predictions as before
df_output = df_pred_sorted[["Race_Level_Race Time"]].copy()
for i in range(1, 21):
    horse_col = f"Horse_{i}_Horse"
    if horse_col in df_pred_sorted.columns:
        df_output[horse_col] = df_pred_sorted[horse_col]
    df_output[f"Horse_{i}_LOG DTW+_PRED"] = preds[:, i - 1].numpy()

df_output.to_csv("predictions/predictions_for_missing_LOGDTW.csv", index=False)
print("Done! Predictions saved in 'predictions/predictions_for_missing_LOGDTW.csv'.")
