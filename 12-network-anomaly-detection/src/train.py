import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
from src.model import BinaryClassifier

# -------------------
# Load data
# -------------------
train_df = pd.read_csv('data/labelled_train.csv')
val_df = pd.read_csv('data/labelled_validation.csv')

X_train = train_df.drop('sus_label', axis=1).values
y_train = train_df['sus_label'].values
X_val = val_df.drop('sus_label', axis=1).values
y_val = val_df['sus_label'].values

# -------------------
# Save features
# -------------------
features = train_df.drop('sus_label', axis=1).columns.tolist()
joblib.dump(features, "models/features.pkl")

# -------------------
# Scale features
# -------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
joblib.dump(scaler, "models/scaler.pkl")

# -------------------
# Convert to tensors
# -------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1,1)

# -------------------
# DataLoaders
# -------------------
batch_size = 4096
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

# -------------------
# Model, loss, optimizer
# -------------------
input_dim = X_train.shape[1]
model = BinaryClassifier(input_dim)
criterion = nn.BCELoss()  # Keep BCELoss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# -------------------
# Training loop
# -------------------
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            val_loss += criterion(model(X_val_batch), y_val_batch).item()
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

# -------------------
# Save model
# -------------------
torch.save(model.state_dict(), "models/model.pt")
print("Model saved.")
