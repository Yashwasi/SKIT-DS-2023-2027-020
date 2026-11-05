import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os

BASE = "reports/raindrop2_analysis"

data = pd.read_csv(
    f"{BASE}/next_minute_pairs.csv"
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        data,
        groups=data["Condition"]
    )
)

train = data.iloc[train_idx].copy()
test = data.iloc[test_idx].copy()

output = f"{BASE}/group_split"

os.makedirs(output, exist_ok=True)

train.to_csv(f"{output}/train.csv", index=False)
test.to_csv(f"{output}/test.csv", index=False)

print("Training samples:", len(train))
print("Testing samples:", len(test))

print("Training conditions:",
      train["Condition"].nunique())

print("Testing conditions:",
      test["Condition"].nunique())

print("Common conditions:",
      len(set(train["Condition"]) & set(test["Condition"])))

print("\nGroup split complete.")