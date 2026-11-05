import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor

BASE = "reports/raindrop2_analysis/splits"
OUTPUT = "reports/raindrop2_analysis/models"

os.makedirs(OUTPUT, exist_ok=True)

train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")

target = "Next_RainRate"

excluded = [
    "D-M-Y", "Time", "Timestamp",
    "Next_Timestamp", "Condition",
    "Rain(mm)", "RainRate(mm/h)", target
]

features = [
    col for col in train.select_dtypes(include="number").columns
    if col not in excluded
]

X_train = train[features].replace([np.inf, -np.inf], np.nan)
X_test = test[features].replace([np.inf, -np.inf], np.nan)

medians = X_train.median()

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, train[target])

predictions = model.predict(X_test)

results = pd.DataFrame({
    "Actual_RainRate": test[target].values,
    "Predicted_RainRate": predictions,
})

results["Absolute_Error"] = abs(
    results["Actual_RainRate"] -
    results["Predicted_RainRate"]
)

results.to_csv(
    f"{OUTPUT}/prediction_comparison.csv",
    index=False
)

plt.figure(figsize=(8, 6))

plt.scatter(
    results["Actual_RainRate"],
    results["Predicted_RainRate"],
    alpha=0.3
)

maximum = max(
    results["Actual_RainRate"].max(),
    results["Predicted_RainRate"].max()
)

plt.plot([0, maximum], [0, maximum], "r--")

plt.xlabel("Actual RainRate (mm/h)")
plt.ylabel("Predicted RainRate (mm/h)")
plt.title("Actual vs Predicted RainRate")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT}/actual_vs_predicted.png",
    dpi=300
)

plt.close()

print("Prediction comparison saved.")
print("Graph saved.")

print("\nSample Predictions:")
print(results.head(10).to_string(index=False))

print("\nMean Absolute Error:",
      round(results["Absolute_Error"].mean(), 4))

print("\nMaximum Absolute Error:",
      round(results["Absolute_Error"].max(), 4))