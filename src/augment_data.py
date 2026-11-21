
import numpy as np
import pandas as pd

df = pd.read_csv("data/raw/cancer_v1.csv")

rng = np.random.default_rng(42)

feature_cols = [c for c in df.columns if c != "target"]

sample = df.sample(20, random_state=42).reset_index(drop=True)

noise = rng.normal(0, 0.05, size=(20, len(feature_cols)))

sample[feature_cols] = sample[feature_cols].values + noise

augmented = pd.concat([df, sample], ignore_index=True)

augmented.to_csv("data/raw/cancer_v1.csv", index=False)

print(f"Dataset now has {len(augmented)} rows")
