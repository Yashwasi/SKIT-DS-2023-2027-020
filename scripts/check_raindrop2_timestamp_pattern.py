import pandas as pd

BASE = "reports/raindrop2_analysis"

df = pd.read_csv(f"{BASE}/splits/test.csv")

pred = pd.read_csv(
    f"{BASE}/models/prediction_comparison.csv"
)

df = df.reset_index(drop=True)

df["Predicted_RainRate"] = pred["Predicted_RainRate"]
df["Absolute_Error"] = pred["Absolute_Error"]

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df["Next_RainRate"] = pd.to_numeric(
    df["Next_RainRate"]
)

df["RainRate(mm/h)"] = pd.to_numeric(
    df["RainRate(mm/h)"]
)

# High prediction errors where actual rainfall is zero
problem = df[
    (df["Next_RainRate"] == 0) &
    (df["Absolute_Error"] > 50)
]

print("\n========== PROBLEM TIMESTAMPS ==========")

summary = problem.groupby("Timestamp").agg(
    Conditions=("Condition", "nunique"),
    Average_Current_Rain=("RainRate(mm/h)", "mean"),
    Average_Predicted=("Predicted_RainRate", "mean"),
    Maximum_Error=("Absolute_Error", "max")
).sort_values("Conditions", ascending=False)

print(summary.head(15).to_string())

print("\n========== 01:30 CHECK ==========")

check = df[
    (df["Timestamp"].dt.strftime("%H:%M") == "01:30") &
    (df["Next_RainRate"] == 0)
]

print("Total matching samples:", len(check))
print("Unique conditions:", check["Condition"].nunique())
print("Average current rainfall:",
      round(check["RainRate(mm/h)"].mean(), 4))
print("Average predicted rainfall:",
      round(check["Predicted_RainRate"].mean(), 4))