"""Legacy exploratory script retained for reference.
The 545-row Housing.csv dataset has been removed from this project.
Use train_estateai_model.py for the active Indian housing workflow.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "dataset" / "Housing.csv")
out = ROOT / "graphs"
out.mkdir(exist_ok=True)

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="area", y="price")
plt.title("Area vs House Price")
plt.tight_layout()
plt.savefig(out / "area_vs_price.png", dpi=150)
plt.close()

numeric = df.select_dtypes(include="number")
plt.figure(figsize=(9, 6))
sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Numerical Feature Correlation")
plt.tight_layout()
plt.savefig(out / "correlation_heatmap.png", dpi=150)
plt.close()
print("EDA graphs saved in the graphs folder.")
