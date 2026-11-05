import pandas as pd
import os

INPUT = "reports/raindrop2_analysis/next_minute_pairs.csv"

OUTPUT = "reports/raindrop2_analysis/splits"

os.makedirs(OUTPUT, exist_ok=True)

data = pd.read_csv(INPUT)

data["Timestamp"] = pd.to_datetime(data["Timestamp"])

data = data.sort_values(["Condition", "Timestamp"])

train_parts = []
test_parts = []

for condition, group in data.groupby("Condition"):

    group = group.sort_values("Timestamp")

    split_index = int(len(group) * 0.70)

    train_parts.append(group.iloc[:split_index])
    test_parts.append(group.iloc[split_index:])

train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

train.to_csv(f"{OUTPUT}/train.csv", index=False)
test.to_csv(f"{OUTPUT}/test.csv", index=False)

print("\n========== DATA SPLIT REPORT ==========")

print("Total samples:", len(data))
print("Training samples:", len(train))
print("Testing samples:", len(test))

print("Training percentage:",
      round(len(train) / len(data) * 100, 2))

print("Testing percentage:",
      round(len(test) / len(data) * 100, 2))

print("\nFiles saved in:", OUTPUT)
print("\n========== COMPLETE ==========")