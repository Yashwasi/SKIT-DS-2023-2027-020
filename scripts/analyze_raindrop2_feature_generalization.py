import pandas as pd
import numpy as np

BASE = "reports/raindrop2_analysis/group_split"

train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")

features = [
    "R-1.2",
    "N",
    "N-1.2",
    "KE(J/m2)",
    "R-1.1",
    "R-1.0",
    "D50",
    "D90",
    "N-1.1",
    "D25"
]

results = []

print("\n========== FEATURE GENERALIZATION ANALYSIS ==========")

for feature in features:

    if feature not in train.columns or feature not in test.columns:
        print(f"Skipping missing feature: {feature}")
        continue

    tr = pd.to_numeric(train[feature], errors="coerce").dropna()
    te = pd.to_numeric(test[feature], errors="coerce").dropna()

    train_min = tr.min()
    train_max = tr.max()

    outside = ((te < train_min) | (te > train_max)).mean() * 100

    train_std = tr.std()

    if train_std > 0:
        mean_difference = (te.mean() - tr.mean()) / train_std
    else:
        mean_difference = np.nan

    results.append({
        "Feature": feature,
        "Train_Mean": tr.mean(),
        "Test_Mean": te.mean(),
        "Train_Std": train_std,
        "Test_Std": te.std(),
        "Train_Min": train_min,
        "Train_Max": train_max,
        "Test_Min": te.min(),
        "Test_Max": te.max(),
        "Test_Outside_Train_Range_Percent": outside,
        "Standardized_Mean_Difference": mean_difference
    })

result = pd.DataFrame(results)

print("\n")
print(result.round(4).to_string(index=False))

output = "reports/raindrop2_analysis/models/feature_generalization.csv"

result.to_csv(output, index=False)

print(f"\nSaved: {output}")
print("\n========== ANALYSIS COMPLETE ==========")