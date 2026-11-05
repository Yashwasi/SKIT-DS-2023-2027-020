import pandas as pd

INPUT = "reports/raindrop2_analysis/models/group_condition_performance.csv"

df = pd.read_csv(INPUT)

df["Category"] = df["Condition"].apply(
    lambda x: "Foliated" if x.startswith("Foliated_")
    else "Unfoliated" if x.startswith("Unfoliated_")
    else "Other"
)

summary = df.groupby("Category").agg(
    Total_Conditions=("Condition", "count"),
    Baseline_MAE=("Baseline_MAE", "mean"),
    RF_MAE=("RF_MAE", "mean"),
    RF_Better=("RF_vs_Baseline", lambda x: (x < 0).sum()),
    RF_Worse=("RF_vs_Baseline", lambda x: (x > 0).sum())
).reset_index()

summary["MAE_Difference"] = (
    summary["RF_MAE"] - summary["Baseline_MAE"]
)

print("\n========== FOLIAGE PERFORMANCE ANALYSIS ==========")
print(summary.round(4).to_string(index=False))

print("\n========== INTERPRETATION ==========")

for _, row in summary.iterrows():
    print(f"\n{row['Category']}:")
    print(f"Conditions: {row['Total_Conditions']}")
    print(f"Baseline MAE: {row['Baseline_MAE']:.4f}")
    print(f"Random Forest MAE: {row['RF_MAE']:.4f}")
    print(f"RF better: {row['RF_Better']}")
    print(f"RF worse: {row['RF_Worse']}")

OUTPUT = "reports/raindrop2_analysis/models/foliage_performance.csv"
summary.to_csv(OUTPUT, index=False)

print(f"\nSaved: {OUTPUT}")