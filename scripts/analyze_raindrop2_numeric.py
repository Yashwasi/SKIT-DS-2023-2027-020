import zipfile
import csv
import math
from io import StringIO

ZIP_PATH = "raindrop2.zip"

feature_values = {}
target_values = []

with zipfile.ZipFile(ZIP_PATH) as z:

    files = [
        f for f in z.namelist()
        if f.lower().endswith(".csv")
    ]

    for file in files:

        content = z.read(file).decode("utf-8-sig")

        lines = [
            line for line in content.splitlines()
            if not line.startswith("#")
        ]

        reader = csv.DictReader(StringIO("\n".join(lines)))

        for row in reader:

            target = row.get("RainRate(mm/h)", "")

            try:
                target = float(target)
            except ValueError:
                target = None

            target_values.append(target)

            for col, value in row.items():

                if col in ["D-M-Y", "Time"]:
                    continue

                try:
                    number = float(value)
                except (ValueError, TypeError):
                    continue

                if not math.isfinite(number):
                    continue

                feature_values.setdefault(col, []).append(number)


print("\n========== NUMERIC FEATURE ANALYSIS ==========")

constant_features = []
variable_features = []

for feature, values in feature_values.items():

    if len(set(values)) <= 1:
        constant_features.append(feature)
    else:
        variable_features.append(feature)

print("Total numeric features:", len(feature_values))
print("Constant features:", len(constant_features))
print("Variable features:", len(variable_features))

print("\n========== CONSTANT FEATURES ==========")

for feature in constant_features:
    print(feature)

print("\n========== TARGET SUMMARY ==========")

valid = [x for x in target_values if x is not None]

print("Valid RainRate:", len(valid))
print("Minimum:", min(valid))
print("Maximum:", max(valid))
print("Mean:", round(sum(valid) / len(valid), 4))

print("\n========== TOP VARIABLE FEATURES ==========")

for feature in variable_features[:20]:
    values = feature_values[feature]

    print(
        feature,
        "| Min:", round(min(values), 4),
        "| Max:", round(max(values), 4)
    )

print("\n========== COMPLETE ==========")