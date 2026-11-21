
from pathlib import Path
from sklearn.datasets import load_breast_cancer

Path("data/raw").mkdir(parents=True, exist_ok=True)

data = load_breast_cancer(as_frame=True)
df = data.frame

df.to_csv("data/raw/cancer_v1.csv", index=False)

print(f"Saved {len(df)} rows to data/raw/cancer_v1.csv")
