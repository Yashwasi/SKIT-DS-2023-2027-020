import pandas as pd
import matplotlib.pyplot as plt

ZIP_DATA = "raindrop2.zip"

# Use the already generated test dataset
df = pd.read_csv(
    "reports/raindrop2_analysis/splits/test.csv"
)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df["RainRate(mm/h)"] = pd.to_numeric(
    df["RainRate(mm/h)"]
)

# Select four problematic dates
dates = [
    "2016-06-16",
    "2016-06-17",
    "2016-06-24",
    "2016-06-28"
]

for date in dates:

    subset = df[
        (df["Timestamp"].dt.strftime("%Y-%m-%d") == date) &
        (df["Timestamp"].dt.strftime("%H:%M") >= "01:20") &
        (df["Timestamp"].dt.strftime("%H:%M") <= "01:40")
    ]

    if subset.empty:
        continue

    grouped = subset.groupby("Timestamp")[
        "RainRate(mm/h)"
    ].mean()

    plt.figure(figsize=(10, 5))

    plt.plot(
        grouped.index,
        grouped.values,
        marker="o"
    )

    plt.axvline(
        pd.Timestamp(date + " 01:30:00"),
        linestyle="--",
        label="01:30 transition"
    )

    plt.title(f"Rainfall Transition - {date}")
    plt.xlabel("Time")
    plt.ylabel("RainRate (mm/h)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"reports/raindrop2_analysis/plots/transition_{date}.png"

    plt.savefig(filename, dpi=200)
    plt.close()

    print("Saved:", filename)

print("\nTransition graphs generated.")