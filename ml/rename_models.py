"""
rename_models.py
----------------
Mevcut uzun isimli pkl dosyalarını kısa hash formatına dönüştürür.
ml/ dizininde çalıştır:  python rename_models.py
"""
import os, re, hashlib

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "outputs", "saved_models")

pattern = re.compile(r"^(.+?)_k(\d+)_([^_]+)_(.+)\.pkl$")

renamed = 0
skipped = 0

for fname in os.listdir(MODELS_DIR):
    if not fname.endswith(".pkl"):
        continue
    m = pattern.match(fname)
    if not m:
        skipped += 1
        continue

    model_name, k, scaler_name, params_str = m.groups()

    # Already short (hash = 8 hex chars, no underscores inside)
    if re.fullmatch(r"[0-9a-f]{8}", params_str):
        skipped += 1
        continue

    params_hash  = hashlib.md5(params_str.encode()).hexdigest()[:8]
    new_name     = f"{model_name}_k{k}_{scaler_name}_{params_hash}.pkl"
    old_path     = os.path.join(MODELS_DIR, fname)
    new_path     = os.path.join(MODELS_DIR, new_name)

    if os.path.exists(new_path):
        os.remove(old_path)
        print(f"[SKIP-DUP] {fname}")
    else:
        os.rename(old_path, new_path)
        print(f"[OK] {fname}\n  -> {new_name}")
    renamed += 1

print(f"\nDone. Renamed: {renamed}, Skipped: {skipped}")