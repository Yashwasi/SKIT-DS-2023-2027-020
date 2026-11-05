import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

INPUT = "reports/raindrop2_analysis/models/group_condition_performance.csv"

df = pd.read_csv(INPUT)

df["Category"] = df["Condition"].apply(
    lambda x: "Foliated" if x.startswith("Foliated_")
    else "Unfoliated" if x.startswith("Unfoliated_")
    else "Other"
)

df = df[df["Category"].isin(["Foliated", "Unfoliated"])].copy()

# Positive difference means RF has higher error.
df["Difference"] = df["RF_MAE"] - df["Baseline_MAE"]

print("\n========== WILCOXON SIGNED-RANK TEST ==========")

for category in ["Foliated", "Unfoliated"]:

    values = df.loc[df["Category"] == category, "Difference"]

    statistic, p_value = wilcoxon(values)

    print(f"\n{category}")
    print("Conditions:", len(values))
    print("Median MAE Difference:", round(values.median(), 4))
    print("Wilcoxon Statistic:", round(statistic, 4))
    print("P-value:", round(p_value, 6))

    if p_value < 0.05:
        print("Result: Statistically significant difference")
    else:
        print("Result: No statistically significant difference")


print("\n========== MANN-WHITNEY U TEST ==========")

foliated = df.loc[
    df["Category"] == "Foliated", "Difference"
]

unfoliated = df.loc[
    df["Category"] == "Unfoliated", "Difference"
]

statistic, p_value = mannwhitneyu(
    foliated,
    unfoliated,
    alternative="two-sided"
)

print("U Statistic:", round(statistic, 4))
print("P-value:", round(p_value, 6))

if p_value < 0.05:
    print("Result: Significant difference between categories")
else:
    print("Result: No significant difference between categories")

print("\nSignificance level: 0.05")
print("\n========== ANALYSIS COMPLETE ==========")