import os
import warnings

import joblib

warnings.filterwarnings("ignore")

folder = "app/saved_models"
files = [f for f in os.listdir(folder) if f.endswith(".pkl")]

for f in files:
    path = os.path.join(folder, f)
    try:
        m = joblib.load(path)
        joblib.dump(m, path)
        print(f"OK: {f}")
    except Exception as e:
        print(f"SKIP: {f} — {e}")
