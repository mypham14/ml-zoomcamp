import sys
import os
import streamlit as st
import torch
import joblib
import numpy as np
import traceback
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import BinaryClassifier  
from src.preprocess import preprocess_input, load_features

# Resolve model file paths
models_dir = os.path.join(project_root, 'models')
features_path = os.path.join(models_dir, 'features.pkl')
scaler_path = os.path.join(models_dir, 'scaler.pkl')
model_path = os.path.join(models_dir, 'model.pt')

# Load features and scaler with error handling
try:
    features = joblib.load(features_path)
    scaler = joblib.load(scaler_path)
except Exception:
    st.error("Failed to load features/scaler. See traceback below:")
    st.code(traceback.format_exc())
    st.stop()

# Load model with error handling and support for different saved formats
try:
    input_dim = len(features)
    model = BinaryClassifier(input_dim)
    state = torch.load(model_path, map_location=torch.device('cpu'))

    # If a full model was saved instead of a state_dict
    if not isinstance(state, dict):
        model = state
    else:
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # Try to adapt keys if they were saved without the 'model.' prefix
            adapted = {}
            for k, v in state.items():
                if not k.startswith('model.'):
                    adapted[f"model.{k}"] = v
                else:
                    adapted[k] = v
            model.load_state_dict(adapted)
    model.eval()
except Exception:
    st.error("Failed to load model. See traceback below:")
    st.code(traceback.format_exc())
    st.stop()


# Streamlit UI
st.title("Cyber Threat Detection")
st.write("Enter the feature values below to predict if the event is suspicious:")


# Load example stats (cached) and show an expander table of stats
@st.cache_data
def load_feature_stats(path, features):
    try:
        df = pd.read_csv(path, usecols=features)
    except Exception:
        return None
    stats = df.agg(['mean', 'median', 'min', 'max']).T
    return stats

train_csv = os.path.join(project_root, 'data', 'labelled_train.csv')
feature_stats = load_feature_stats(train_csv, features)

if feature_stats is not None:
    with st.expander("Show example feature stats (mean, median, min, max)", expanded=False):
        st.dataframe(feature_stats.round(4))
else:
    st.info("No training data found for example feature values.")

# Collect user input with grouped layout and validation
input_dict = {}
st.write("### Input features")
col_count = 3
cols = st.columns(col_count)

# Buttons for presets
preset_col = st.columns([1])[0]
if preset_col.button("Use mean defaults") and feature_stats is not None:
    preset_defaults = {f: float(feature_stats.loc[f,'mean']) for f in features}
    # set defaults by re-rendering inputs with these values via session state (clamped to min/max)
    for k, v in preset_defaults.items():
        key = f"pref_{k}"
        if feature_stats is not None and k in feature_stats.index:
            min_v = float(feature_stats.loc[k, 'min'])
            max_v = float(feature_stats.loc[k, 'max'])
            val = max(min_v, min(round(v, 3), max_v))
        else:
            val = round(v, 3)
        st.session_state[key] = val

# Render inputs in columns with min/max from feature_stats if available
for i, f in enumerate(features):
    col = cols[i % col_count]
    default_val = None
    min_val = None
    max_val = None
    help_text = None
    if feature_stats is not None and f in feature_stats.index:
        stats = feature_stats.loc[f]
        default_val = round(float(stats['mean']), 3)
        min_val = float(stats['min'])
        max_val = float(stats['max'])
        help_text = f"Example (mean): {default_val}. Range: [{min_val}, {max_val}]"
    key = f"pref_{f}"
    # Determine initial value and ensure session state is set before creating widget
    if key in st.session_state:
        # Use existing session state value (don't pass 'value' to widget to avoid Streamlit warning)
        if min_val is not None and max_val is not None:
            try:
                st.session_state[key] = max(min_val, min(float(st.session_state[key]), max_val))
            except Exception:
                st.session_state[key] = default_val if default_val is not None else 0.0
            input_dict[f] = col.number_input(f"{f}:", min_value=min_val, max_value=max_val, help=help_text, key=key)
        else:
            input_dict[f] = col.number_input(f"{f}:", help=help_text, key=key)
    else:
        # Key not present in session state; compute init and pass it as widget default
        init = default_val if default_val is not None else 0.0
        if min_val is not None and max_val is not None:
            try:
                init = max(min_val, min(float(init), max_val))
            except Exception:
                init = default_val if default_val is not None else min_val
            input_dict[f] = col.number_input(f"{f}:", value=init, min_value=min_val, max_value=max_val, help=help_text, key=key)
        else:
            input_dict[f] = col.number_input(f"{f}:", value=init, help=help_text, key=key)

# Predict
predict_btn = st.button("Predict")

if predict_btn:
    # Convert input to array
    x = np.array([input_dict[f] for f in features]).reshape(1, -1)
    # Scale input
    x_scaled = scaler.transform(x)
    # Convert to tensor
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    # Predict (model already has Sigmoid)
    prob = model(x_tensor).item()
    label = int(prob >= 0.5)

    st.write(f"Suspicious Probability: {prob*100:.2f}%")
    st.write("Prediction:", "Suspicious" if label == 1 else "Normal")
