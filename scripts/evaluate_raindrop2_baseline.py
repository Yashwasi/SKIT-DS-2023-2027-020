import pandas as pd
import numpy as np

FILE = "reports/raindrop2_analysis/next_minute_pairs.csv"

data = pd.read_csv(FILE)

actual = data["Next_RainRate"].to_numpy()
predicted = data["RainRate(mm/h)"].to_numpy()

# Evaluation metrics
mae = np.mean(np.abs(actual - predicted))

rmse = np.sqrt(np.mean((actual - predicted) ** 2))

errors = actual - predicted

# R2 score
ss_res = np.sum(errors ** 2)
ss_tot = np.sum((actual - actual.mean()) ** 2)

r2 = 1 - (ss_res / ss_tot)

print("\n========== PERSISTENCE BASELINE ==========")

print("Total samples:", len(data))

print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

print("\n========== COMPLETE ==========")