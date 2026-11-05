import pandas as pd
import numpy as np
import os
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE = "reports/raindrop2_analysis/group_split"
OUTPUT = "reports/raindrop2_analysis/models"

os.makedirs(OUTPUT, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading group split datasets...")

train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")

target = "Next_RainRate"

print("Training rows:", len(train))
print("Testing rows:", len(test))

# ==========================================
# 3. FEATURE SELECTION
# ==========================================

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

print("\nTotal features:", len(features))

# ==========================================
# 4. DATA PREPROCESSING
# ==========================================

X_train = train[features].replace(
    [np.inf, -np.inf], np.nan
)

X_test = test[features].replace(
    [np.inf, -np.inf], np.nan
)

y_train = train[target]
y_test = test[target]

# Training median imputation
medians = X_train.median()

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

# ==========================================
# 5. RANDOM FOREST MODEL
# ==========================================

print("\nTraining Group Random Forest...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

start = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start

# ==========================================
# 6. PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# 7. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n========== GROUP RANDOM FOREST RESULTS ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features:", len(features))

print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

print("Training time:", round(training_time, 2), "seconds")

# ==========================================
# 8. SAVE PREDICTIONS
# ==========================================

prediction_df = pd.DataFrame({
    "Actual_RainRate": y_test.values,
    "Predicted_RainRate": predictions,
    "Absolute_Error": np.abs(
        y_test.values - predictions
    )
})

prediction_df.to_csv(
    f"{OUTPUT}/group_prediction_comparison.csv",
    index=False
)

print("\nPredictions saved successfully.")

# ==========================================
# 9. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

importance.to_csv(
    f"{OUTPUT}/group_randomforest_importance.csv",
    index=False
)

print("\n========== TOP 15 GROUP MODEL FEATURES ==========")

print(
    importance.head(15).to_string(index=False)
)

print("\nFeature importance saved successfully.")

# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n========== OUTPUT FILES ==========")

print("1. group_prediction_comparison.csv")
print("2. group_randomforest_importance.csv")

print("\n========== ANALYSIS COMPLETE ==========")
