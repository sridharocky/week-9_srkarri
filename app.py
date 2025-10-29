import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.modeling import GroupEstimate  # updated to use your custom model
from utils.visualizations import plot_sentiment_distribution

st.set_page_config(page_title="Coffee Sentiment Analyzer", page_icon="☕", layout="wide")

st.title("☕ Coffee Review Sentiment Analyzer")
st.markdown("Analyze coffee reviews, visualize sentiment, and explore insights interactively.")

st.subheader("📊 Load Data")

# Use a default dataset if secrets are missing
try:
    sheet_url = st.secrets["public_gsheet_url"]
    df = load_data(sheet_url)
except KeyError:
    st.warning("No Google Sheet URL found in secrets. Using example dataset.")
    df = pd.DataFrame({
        "loc_country": ["Guatemala", "Mexico", "Mexico", "Guatemala"],
        "roast": ["Light", "Medium", "Dark", "Light"],
        "rating": [88, 91, 85, 90]
    })

if df.empty:
    st.stop()

# Clean data
df = clean_data(df)
st.dataframe(df.head())

# Fit GroupEstimate model
if {"loc_country", "roast", "rating"}.issubset(df.columns):
    X = df[["loc_country", "roast"]]
    y = df["rating"]
    model = GroupEstimate(estimate="mean")
    model.fit(X, y, default_category="loc_country")
else:
    st.error("Dataset must contain 'loc_country', 'roast', and 'rating' for modeling.")
    st.stop()

# Sentiment visualization
st.subheader("📈 Sentiment Visualization")
if st.button("Show Sentiment Distribution"):
    plot_sentiment_distribution(df)

# Predict example
st.subheader("🔍 Predict Example (Optional)")
if {"loc_country", "roast"}.issubset(df.columns):
    feature1 = st.text_input("Country", value="Guatemala")
    feature2 = st.text_input("Roast", value="Light")
    if st.button("Predict Rating"):
        pred = model.predict([[feature1, feature2]])[0]
        st.success(f"Predicted Rating: {pred}")
else:
    st.info("Add 'loc_country' and 'roast' columns to your dataset to enable predictions.")
