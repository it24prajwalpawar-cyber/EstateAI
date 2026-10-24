"""Legacy exploratory script retained for reference.
The 545-row Housing.csv dataset has been removed from this project.
Use train_estateai_model.py for the active Indian housing workflow.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "dataset" / "Housing.csv")
X = pd.get_dummies(df.drop("price", axis=1), drop_first=True, dtype=int)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Original dataset:", df.shape)
print("Encoded features:", X.shape)
print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])
print("Encoded columns:", X.columns.tolist())
