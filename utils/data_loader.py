import pandas as pd
import streamlit as st

@st.cache_data
def load_data(sheet_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(sheet_url)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()
