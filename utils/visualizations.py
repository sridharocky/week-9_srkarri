import streamlit as st
import matplotlib.pyplot as plt

st.subheader("📈 Sentiment Visualization")

# Only create a single container for the plot
with st.expander("Show Sentiment Distribution"):
    if 'sentiment' in df.columns:
        fig, ax = plt.subplots()
        df['sentiment'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.warning("No 'sentiment' column found in the dataset.")
