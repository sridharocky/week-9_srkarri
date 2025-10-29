import streamlit as st
import matplotlib.pyplot as plt

def plot_sentiment_distribution(df):
    if 'sentiment' not in df.columns:
        st.warning("No 'sentiment' column found in the dataset.")
        return

    if df['sentiment'].dropna().empty:
        st.warning("The 'sentiment' column has no valid data.")
        return

    sentiment_counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sentiment_counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

# --- In app.py ---
st.subheader("📈 Sentiment Visualization")

# Initialize session state for the button
if 'show_plot' not in st.session_state:
    st.session_state.show_plot = False

# Button click updates the session state
if st.button("Show Sentiment Distribution"):
    st.session_state.show_plot = True

# Only show the plot if button was clicked
if st.session_state.show_plot:
    plot_sentiment_distribution(df)
