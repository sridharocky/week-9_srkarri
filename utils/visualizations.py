# utils/visualizations.py
import streamlit as st
import matplotlib.pyplot as plt

def plot_sentiment_distribution(df):
    """Plot sentiment distribution derived from 'rating' column."""
    if 'rating' not in df.columns:
        st.warning("No 'rating' column found in the dataset.")
        return

    # Create sentiment column
    def rating_to_sentiment(r):
        if r >= 90:
            return "positive"
        elif r >= 80:
            return "neutral"
        else:
            return "negative"

    df['sentiment'] = df['rating'].apply(rating_to_sentiment)

    # Plot
    fig, ax = plt.subplots()
    df['sentiment'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)
