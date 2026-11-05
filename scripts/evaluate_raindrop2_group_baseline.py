import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE = "reports/raindrop2_analysis/group_split"
MODEL = "reports/raindrop2_analysis/models"

test = pd.read_csv(f"{BASE}/test.csv")

rf = pd.read_csv(
    f"{MODEL}/group_prediction_comparison.csv"
)

actual = test["Next_RainRate"].values

# Persistence baseline
baseline = test["RainRate(mm/h)"].values

# Random Forest
rf_predictions = rf["Predicted_RainRate"].values

def evaluate(name, actual, predictions):

    print(f"\n{name}")

    print("MAE:",
          round(mean_absolute_error(actual, predictions), 4))

    print("RMSE:",
          round(np.sqrt(mean_squared_error(actual, predictions)), 4))

    print("R2:",
          round(r2_score(actual, predictions), 4))


print("\n========== GROUP TEST COMPARISON ==========")

evaluate("PERSISTENCE BASELINE", actual, baseline)

evaluate("RANDOM FOREST", actual, rf_predictions)

print("\n========== COMPLETE ==========")