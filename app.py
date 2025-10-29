import streamlit as st
import pandas as pd
import pickle
from utils.data_loader import load_data
from utils.preprocessing import clean_data

st.title("☕ Coffee Review Estimator")

# Load raw data
sheet_url = st.secrets["public_gsheet_url"]
df_raw = load_data(sheet_url)
df_clean = clean_data(df_raw)
st.dataframe(df_clean.head())

# Load model
with open('model/group_estimate.pkl', 'rb') as f:
    gm = pickle.load(f)

st.subheader("Predict Coffee Rating")
country = st.selectbox("Country", options=df_clean['loc_country'].unique())
roast = st.selectbox("Roast Type", options=df_clean['roast'].unique())

if st.button("Predict"):
    prediction = gm.predict([[country, roast]])
    st.write(f"Predicted Rating: {prediction[0]}")
