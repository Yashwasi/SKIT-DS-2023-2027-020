import pandas as pd

BASE = "reports/raindrop2_analysis/splits"

train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")

def audit(name, df):

    target = pd.to_numeric(
        df["Next_RainRate"],
        errors="coerce"
    )

    print(f"\n========== {name} DATA ==========")

    print("Total samples:", len(df))
    print("Zero rainfall targets:", (target == 0).sum())
    print("Non-zero rainfall targets:", (target > 0).sum())

    print(
        "Zero rainfall percentage:",
        round((target == 0).mean() * 100, 2),
        "%"
    )

    print("Current timestamp range:")
    print(df["Timestamp"].min(), "to", df["Timestamp"].max())

    print("\nTarget distribution by next timestamp:")

    timestamps = pd.to_datetime(
        df["Next_Timestamp"],
        errors="coerce"
    )

    summary = pd.DataFrame({
        "Minute": timestamps.dt.strftime("%H:%M"),
        "Target": target
    })

    print(
        summary.groupby("Minute")["Target"]
        .agg(
            Total="count",
            Zero_Rainfall=lambda x: (x == 0).sum()
        )
        .to_string()
    )


audit("TRAIN", train)
audit("TEST", test)

print("\n========== AUDIT COMPLETE ==========")