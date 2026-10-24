"""Register the dataset: confirm it exists in the repo, has the expected columns, and print a summary."""
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

df = pd.read_csv(DATA_PATH)

missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    sys.exit(f"ERROR: dataset is missing expected columns: {missing}")

print(f"Dataset registered from {DATA_PATH}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Duplicate rows: {df.duplicated().sum()}")
nulls = df.isna().sum()
print("Columns with missing values:", nulls[nulls > 0].to_dict() or "none")
print("Target distribution (ProdTaken):", df["ProdTaken"].value_counts(normalize=True).round(3).to_dict())
