import pandas as pd
from apputil import GroupEstimate

# Sample data
df_raw = pd.DataFrame({
    "loc_country": ["Guatemala", "Mexico", "Brazil", "Guatemala", "Mexico"],
    "roast": ["Light", "Medium", "Dark", "Light", "Medium"],
    "rating": [88, 91, 85, 89, 90]
})

X = df_raw[["loc_country", "roast"]]
y = df_raw["rating"]

print("=== Mean Estimate ===")
gm = GroupEstimate(estimate="mean")
gm.fit(X, y)
X_new = [["Guatemala", "Light"], ["Mexico", "Medium"], ["Canada", "Dark"]]
print(gm.predict(X_new))   # -> [88.5, 90.5, nan]

print("\n=== Mean with Default Category (loc_country) ===")
gm.fit(X, y, default_category="loc_country")
print(gm.predict(X_new))   # -> [88.5, 90.5, nan] (fallback handled)
