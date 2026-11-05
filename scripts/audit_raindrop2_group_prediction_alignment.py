import pandas as pd
import numpy as np

BASE = "reports/raindrop2_analysis/group_split"
MODEL = "reports/raindrop2_analysis/models"

test = pd.read_csv(f"{BASE}/test.csv")
pred = pd.read_csv(
    f"{MODEL}/group_prediction_comparison.csv"
)

print("\n========== PREDICTION ALIGNMENT AUDIT ==========")

print("\n1. ROW COUNT")
print("Test rows:", len(test))
print("Prediction rows:", len(pred))

if len(test) != len(pred):
    raise ValueError("Row count mismatch!")

print("\n2. REQUIRED COLUMNS")

required_test = [
    "Condition",
    "Next_Timestamp",
    "Next_RainRate"
]

required_pred = [
    "Predicted_RainRate"
]

for col in required_test:
    print(f"{col}: {'Present' if col in test.columns else 'MISSING'}")

for col in required_pred:
    print(f"{col}: {'Present' if col in pred.columns else 'MISSING'}")

if not all(c in test.columns for c in required_test):
    raise ValueError("Required test columns missing!")

if not all(c in pred.columns for c in required_pred):
    raise ValueError("Prediction column missing!")

print("\n3. PREDICTION VALUE CHECK")

print("Missing predictions:", pred["Predicted_RainRate"].isna().sum())

print(
    "Infinite predictions:",
    np.isinf(pred["Predicted_RainRate"]).sum()
)

print(
    "Negative predictions:",
    (pred["Predicted_RainRate"] < 0).sum()
)

print("\n4. ROW-WISE ALIGNMENT")

if "Condition" in pred.columns:
    print(
        "Condition match:",
        (test["Condition"] == pred["Condition"]).all()
    )
else:
    print("Prediction file has no Condition column.")

if "Next_RainRate" in pred.columns:
    print(
        "Actual target match:",
        np.allclose(
            test["Next_RainRate"],
            pred["Next_RainRate"],
            equal_nan=True
        )
    )
else:
    print("Prediction file has no actual target column.")

print("\n5. SAMPLE COMPARISON")

comparison = pd.DataFrame({
    "Condition": test["Condition"],
    "Timestamp": test["Next_Timestamp"],
    "Actual": test["Next_RainRate"],
    "Prediction": pred["Predicted_RainRate"]
})

print(comparison.head(10).to_string(index=False))

print("\n6. FINAL CHECK")
print("Audit completed.")