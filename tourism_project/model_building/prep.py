"""Data preparation: load -> clean -> train/test split -> save CSVs (passed on as workflow artifacts)."""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# 1. Remove unnecessary columns (the unnamed index column and the customer identifier carry no signal)
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

# 2. Fix inconsistent category label ('Fe Male' is a typo for 'Female')
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# 3. Drop exact duplicates
df = df.drop_duplicates().reset_index(drop=True)
print("Clean shape:", df.shape)

# 4. Stratified train/test split (the target is imbalanced, ~19% buyers)
X, y = df.drop(columns=[TARGET]), df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print(f"Saved splits -> train: {Xtrain.shape}, test: {Xtest.shape}")
