import streamlit as st
from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.modeling import load_model
from utils.visualizations import plot_sentiment_distribution

# Page config
st.set_page_config(page_title="Coffee Sentiment Analyzer", page_icon="☕", layout="wide")

st.title("☕ Coffee Review Sentiment Analyzer")
st.markdown("Analyze coffee reviews, visualize sentiment, and explore insights interactively.")

# --------------------------
# Load Data
# --------------------------
st.subheader("📊 Load Data")

# Ensure you have the secret 'public_gsheet_url' set in Streamlit secrets
try:
    sheet_url = st.secrets["public_gsheet_url"]
except KeyError:
    st.error("Missing 'public_gsheet_url' in Streamlit secrets. Please add it to run the app.")
    st.stop()

df = load_data(sheet_url)
if df.empty:
    st.warning("No data found in the Google Sheet.")
    st.stop()

df = clean_data(df)
st.dataframe(df.head())

# --------------------------
# Load Model
# --------------------------
model = load_model()

# --------------------------
# Sentiment Visualization
# --------------------------
st.subheader("📈 Sentiment Visualization")
if st.button("Show Sentiment Distribution", key="sentiment_dist"):
    plot_sentiment_distribution(df)

# --------------------------
# Predict Example
# --------------------------
st.subheader("🔍 Predict Example (Optional)")
if {'feature1', 'feature2'}.issubset(df.columns):
    feature1 = st.number_input("Feature 1")
    feature2 = st.number_input("Feature 2")
    if st.button("Predict Sentiment", key="predict_sentiment"):
        pred = model.predict([[feature1, feature2]])[0]
        st.success(f"Predicted Sentiment: {pred}")
else:
    st.info("Add 'feature1' and 'feature2' to your dataset to enable predictions.")
