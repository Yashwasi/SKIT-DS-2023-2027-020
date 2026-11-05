import pandas as pd

BASE = "reports/raindrop2_analysis/group_split"
MODEL = "reports/raindrop2_analysis/models"

test = pd.read_csv(f"{BASE}/test.csv")
rf = pd.read_csv(f"{MODEL}/group_prediction_comparison.csv")

test = test.reset_index(drop=True)
rf = rf.reset_index(drop=True)

test["RF_Prediction"] = rf["Predicted_RainRate"]

test["Baseline_Error"] = (
    test["Next_RainRate"] - test["RainRate(mm/h)"]
).abs()

test["RF_Error"] = (
    test["Next_RainRate"] - test["RF_Prediction"]
).abs()

result = test.groupby("Condition").agg(
    Samples=("Next_RainRate", "count"),
    Baseline_MAE=("Baseline_Error", "mean"),
    RF_MAE=("RF_Error", "mean"),
    Actual_Mean=("Next_RainRate", "mean")
).reset_index()

result["RF_vs_Baseline"] = (
    result["RF_MAE"] - result["Baseline_MAE"]
)

result = result.sort_values("RF_vs_Baseline")

print("\n========== TOP 10 CONDITIONS: RF BETTER ==========")
print(result.head(10).to_string(index=False))

print("\n========== TOP 10 CONDITIONS: RF WORSE ==========")
print(
    result.tail(10)
    .sort_values("RF_vs_Baseline", ascending=False)
    .to_string(index=False)
)

result.to_csv(
    f"{MODEL}/group_condition_performance.csv",
    index=False
)

print("\nSaved: group_condition_performance.csv")