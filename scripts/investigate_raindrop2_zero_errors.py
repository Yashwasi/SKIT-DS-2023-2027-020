import pandas as pd
import numpy as np

BASE = "reports/raindrop2_analysis"

test = pd.read_csv(f"{BASE}/splits/test.csv")
pred = pd.read_csv(
    f"{BASE}/models/prediction_comparison.csv"
)

df = test.reset_index(drop=True).copy()

df["Predicted_RainRate"] = pred["Predicted_RainRate"]
df["Absolute_Error"] = pred["Absolute_Error"]

zero = df[df["Next_RainRate"] == 0].copy()

zero = zero.sort_values(
    "Absolute_Error", ascending=False
)

print("Total zero-rainfall test samples:", len(zero))

print("\n========== WORST 15 ZERO-RAINFALL CASES ==========")

columns = [
    "Condition",
    "Timestamp",
    "RainRate(mm/h)",
    "Next_RainRate",
    "Predicted_RainRate",
    "Absolute_Error"
]

print(zero[columns].head(15).to_string(index=False))

print("\n========== ZERO-RAINFALL SUMMARY ==========")

print("Average prediction:",
      round(zero["Predicted_RainRate"].mean(), 4))

print("Average absolute error:",
      round(zero["Absolute_Error"].mean(), 4))

print("Maximum prediction:",
      round(zero["Predicted_RainRate"].max(), 4))