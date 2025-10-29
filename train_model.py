import pandas as pd
from utils.modeling import GroupEstimate
import pickle
from utils.data_loader import load_data
from utils.preprocessing import clean_data

# Load and clean data
sheet_url = "https://docs.google.com/spreadsheets/d/e/XXXXXX/pub?output=csv"
df_raw = load_data(sheet_url)
df_clean = clean_data(df_raw)

X = df_clean[['loc_country', 'roast']]
y = df_clean['rating']

gm = GroupEstimate(estimate='mean')
gm.fit(X, y, default_category='loc_country')

# Save the trained model
with open('model/group_estimate.pkl', 'wb') as f:
    pickle.dump(gm, f)
print("✅ GroupEstimate model saved.")
