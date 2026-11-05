import pandas as pd

BASE = "reports/raindrop2_analysis/group_split"

train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")

features = [
    "R-1.2",
    "N",
    "N-1.2",
    "KE(J/m2)",
    "R-1.1",
    "R-1.0",
    "D50",
    "D90",
    "N-1.1",
    "D25"
]

results = []

for feature in features:

    train_corr = train[feature].corr(
        train["Next_RainRate"]
    )

    test_corr = test[feature].corr(
        test["Next_RainRate"]
    )

    results.append({
        "Feature": feature,
        "Train_Correlation": train_corr,
        "Test_Correlation": test_corr,
        "Correlation_Difference": test_corr - train_corr
    })

result = pd.DataFrame(results)

print("\n========== FEATURE-TARGET RELATIONSHIP SHIFT ==========")
print(result.round(4).to_string(index=False))

output = (
    "reports/raindrop2_analysis/models/"
    "feature_target_shift.csv"
)

result.to_csv(output, index=False)

print(f"\nSaved: {output}")
print("\n========== COMPLETE ==========")