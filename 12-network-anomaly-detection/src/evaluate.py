import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from src.model import BinaryClassifier

# Load model
features = joblib.load("models/features.pkl")
scaler = joblib.load("models/scaler.pkl")
input_dim = len(features)
model = BinaryClassifier(input_dim)
model.load_state_dict(torch.load("models/model.pt", map_location=torch.device('cpu')))
model.eval()

# Load datasets
train_df = pd.read_csv("data/labelled_train.csv")
val_df = pd.read_csv("data/labelled_validation.csv")
test_df = pd.read_csv("data/labelled_test.csv")

# Combine all
for df, split in zip([train_df, val_df, test_df], ["train","validation","test"]):
    df["split"] = split

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Prepare features
X_all = all_df.drop(["sus_label","split"], axis=1)
X_all_scaled = scaler.transform(X_all)
X_all_tensor = torch.tensor(X_all_scaled, dtype=torch.float32)

# Predictions
with torch.no_grad():
    probs = model(X_all_tensor).squeeze().numpy()
    preds = (probs >= 0.5).astype(int)

all_df["predicted_label"] = preds
all_df["prediction_probability"] = probs
all_df["prediction_correct"] = (all_df["predicted_label"] == all_df["sus_label"]).astype(int)

# Save predictions
all_df.to_csv("all_predictions.csv", index=False)
print("Predictions saved.")

# Metrics for validation and test sets
for split in ["validation","test"]:
    subset = all_df[all_df["split"]==split]
    print(f"\n--- {split.upper()} ---")
    print("Confusion Matrix:")
    print(confusion_matrix(subset["sus_label"], subset["predicted_label"]))
    print("Classification Report:")
    print(classification_report(subset["sus_label"], subset["predicted_label"], target_names=["Normal","Suspicious"]))
