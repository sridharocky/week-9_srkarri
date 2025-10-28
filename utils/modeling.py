import pickle
from sklearn.linear_model import LogisticRegression

def train_model(X, y, model_path="model/model.pkl"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print("✅ Model saved to:", model_path)
    return model

def load_model(model_path="model/model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
