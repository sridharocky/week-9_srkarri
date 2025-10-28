from utils.data_loader import load_data
from utils.preprocessing import clean_data
from utils.modeling import train_model

sheet_url = "https://docs.google.com/spreadsheets/d/e/XXXXXX/pub?output=csv"

print("📥 Loading data...")
df = load_data(sheet_url)
df_clean = clean_data(df)

if {'feature1', 'feature2', 'target'}.issubset(df_clean.columns):
    X = df_clean[['feature1', 'feature2']]
    y = df_clean['target']
    print("🧠 Training model...")
    train_model(X, y)
else:
    print("⚠️ Dataset must contain 'feature1', 'feature2', and 'target' columns.")
