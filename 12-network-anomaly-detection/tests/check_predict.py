import os
import sys
import joblib
import torch
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import BinaryClassifier

models_dir = os.path.join(project_root, 'models')
features = joblib.load(os.path.join(models_dir, 'features.pkl'))
scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
state = torch.load(os.path.join(models_dir, 'model.pt'), map_location='cpu')

m = BinaryClassifier(len(features))
if not isinstance(state, dict):
    m = state
else:
    try:
        m.load_state_dict(state)
    except RuntimeError:
        adapted = {}
        for k, v in state.items():
            if not k.startswith('model.'):
                adapted[f"model.{k}"] = v
            else:
                adapted[k] = v
        m.load_state_dict(adapted)

m.eval()

x = np.zeros((1, len(features)))
x_scaled = scaler.transform(x)
x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
prob = m(x_tensor).item()
print('probability:', prob)