import streamlit as st
import matplotlib.pyplot as plt

def plot_sentiment_distribution(df):
    if 'sentiment' not in df.columns:
        st.warning("No 'sentiment' column found in the dataset.")
        return

    # Count the sentiment values
    sentiment_counts = df['sentiment'].value_counts()

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sentiment_counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Render the figure in Streamlit
    st.pyplot(fig)
    plt.close(fig)  # Close figure to avoid overlapping plots in repeated runs
