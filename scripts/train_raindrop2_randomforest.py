import pandas as pd
import numpy as np
import os
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = "reports/raindrop2_analysis/splits"

train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")

target = "Next_RainRate"

excluded = [
    "D-M-Y",
    "Time",
    "Timestamp",
    "Next_Timestamp",
    "Condition",
    "Rain(mm)",
    "RainRate(mm/h)",
    target
]

features = [
    col for col in train.select_dtypes(include="number").columns
    if col not in excluded
]

X_train = train[features].replace([np.inf, -np.inf], np.nan)
X_test = test[features].replace([np.inf, -np.inf], np.nan)

y_train = train[target]
y_test = test[target]

# Handle missing values using training medians
medians = X_train.median()

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Selected features:", len(features))

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

start = time.time()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n========== RANDOM FOREST RESULTS ==========")

print("MAE:", round(mean_absolute_error(y_test, predictions), 4))

print("RMSE:", round(
    np.sqrt(mean_squared_error(y_test, predictions)), 4
))

print("R2 Score:", round(r2_score(y_test, predictions), 4))

print("Training time:", round(time.time() - start, 2), "seconds")

# Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

os.makedirs("reports/raindrop2_analysis/models", exist_ok=True)

importance.to_csv(
    "reports/raindrop2_analysis/models/randomforest_importance.csv",
    index=False
)

print("\nTop 10 important features:")
print(importance.head(10).to_string(index=False))

print("\n========== COMPLETE ==========")