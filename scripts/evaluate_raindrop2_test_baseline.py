import pandas as pd
import numpy as np
import os

TEST_FILE = "reports/raindrop2_analysis/splits/test.csv"

OUTPUT_DIR = "reports/raindrop2_analysis"

data = pd.read_csv(TEST_FILE)

actual = data["Next_RainRate"].to_numpy()
predicted = data["RainRate(mm/h)"].to_numpy()

mae = np.mean(np.abs(actual - predicted))

rmse = np.sqrt(np.mean((actual - predicted) ** 2))

ss_res = np.sum((actual - predicted) ** 2)
ss_tot = np.sum((actual - actual.mean()) ** 2)

r2 = 1 - ss_res / ss_tot

print("\n========== TEST BASELINE RESULTS ==========")

print("Test samples:", len(data))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

print("\n========== COMPLETE ==========")