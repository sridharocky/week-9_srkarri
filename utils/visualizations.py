import streamlit as st
import matplotlib.pyplot as plt

def plot_sentiment_distribution(df):
    if 'sentiment' not in df.columns:
        st.warning("No 'sentiment' column found in the dataset.")
        return

    fig, ax = plt.subplots()
    df['sentiment'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)
