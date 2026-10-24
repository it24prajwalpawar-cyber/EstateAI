"""Legacy exploratory script retained for reference.
The 545-row Housing.csv dataset has been removed from this project.
Use train_estateai_model.py for the active Indian housing workflow.
"""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
dataset = ROOT / "dataset" / "Housing.csv"
dataset.parent.mkdir(parents=True, exist_ok=True)
url = "https://raw.githubusercontent.com/MartinJessenne/Housing-Price-Prediction-Kaggle-Dataset/main/Housing.csv"
if not dataset.exists():
    print("Downloading Housing.csv...")
    urlretrieve(url, dataset)

df = pd.read_csv(dataset)
print("First five rows:")
print(df.head())
print("\\nDataset shape:", df.shape)
print("\\nColumn information:")
df.info()
print("\\nMissing values:")
print(df.isnull().sum())
