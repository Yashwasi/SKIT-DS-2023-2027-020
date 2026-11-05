import pandas as pd

FILE = "reports/raindrop2_analysis/models/prediction_comparison.csv"

df = pd.read_csv(FILE)

df["Actual_RainRate"] = pd.to_numeric(
    df["Actual_RainRate"]
)

df["Predicted_RainRate"] = pd.to_numeric(
    df["Predicted_RainRate"]
)

df["Absolute_Error"] = pd.to_numeric(
    df["Absolute_Error"]
)

print("\n========== WORST 15 PREDICTIONS ==========")

worst = df.nlargest(15, "Absolute_Error")

print(worst.to_string(index=False))

print("\n========== ERROR STATISTICS ==========")

print("90th percentile:",
      round(df["Absolute_Error"].quantile(0.90), 4))

print("95th percentile:",
      round(df["Absolute_Error"].quantile(0.95), 4))

print("99th percentile:",
      round(df["Absolute_Error"].quantile(0.99), 4))

print("\n========== HIGH RAINFALL PERFORMANCE ==========")

high = df[df["Actual_RainRate"] >= 100]

print("High rainfall samples:", len(high))

if len(high) > 0:
    print("High rainfall MAE:",
          round(high["Absolute_Error"].mean(), 4))