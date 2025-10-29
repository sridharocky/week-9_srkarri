import os
import pandas as pd
import pickle
from utils.modeling import GroupEstimate
from utils.data_loader import load_data
from utils.preprocessing import clean_data

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Load and clean data
sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRFcfzpMx7fsoBrwSlwlM7ShPiD5EoO0NiBuSRzQNkn8X6PA1pPLXyJj1DIoOvZAka9FD60MMxiNK8w/pub?gid=81999126&single=true&output=csv"
df_raw = load_data(sheet_url)
df_clean = clean_data(df_raw)

# Ensure all column names are strings, then normalize
df_clean.columns = df_clean.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")

# Print column names to debug
print("Columns after cleaning:", df_clean.columns.tolist())
print("Sample data:\n", df_clean.head())

# Ensure required columns exist
required_columns = ['loc_country', 'roast', 'rating']
missing_columns = [col for col in required_columns if col not in df_clean.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in data: {missing_columns}")

# Drop rows with missing values in features or target
df_clean = df_clean.dropna(subset=['loc_country', 'roast', 'rating'])

# Select features and target
X = df_clean[['loc_country', 'roast']]
y = df_clean['rating']

# Train GroupEstimate model
gm = GroupEstimate(estimate='mean')
gm.fit(X, y, default_category='loc_country')

# Save trained model
model_path = "model/group_estimate.pkl"
with open(model_path, "wb") as f:
    pickle.dump(gm, f)

print(f"✅ GroupEstimate model saved at: {model_path}")
