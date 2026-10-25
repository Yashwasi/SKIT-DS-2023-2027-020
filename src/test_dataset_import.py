
from pathlib import Path
import pandas as pd

# Project root is the parent of the src folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "raindrop_volume_processed.zip"

print("Dataset path:", ZIP_PATH)
print("Dataset file exists:", ZIP_PATH.exists())

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {ZIP_PATH}")

# Read only 10,000 rows to test the import
df = pd.read_csv(
    ZIP_PATH,
    compression="zip",
    nrows=10_000
)

print("\nDataset imported successfully!")
print("Sample shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values in this sample:")
print(df.isna().sum())
