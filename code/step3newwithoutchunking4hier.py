import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import gc
import psutil
import argparse

# === Parse command-line arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input CSV file with k-mer features")
parser.add_argument("--output", required=True, help="Output CSV file for scaled features")
parser.add_argument("--importance_out", required=True, help="Output CSV file for feature importances")
args = parser.parse_args()
print(f"📥 step3newwithoutchunking.py → Input: {args.input}, Scaled Output: {args.output}, Importance Output: {args.importance_out}")


print("🔄 Loading dataset...")
df = pd.read_csv(args.input)

# === Step 1: Separate features and labels efficiently ===
labels = df["Label"].values  # handle label column separately
X = df.drop(columns=["Label"]).astype(np.float32).values  # convert to float32
del df  # free memory
gc.collect()

# === Step 2: Remove zero-variance features ===
print("🧹 Removing zero-variance features...")
var_mask = np.var(X, axis=0) > 0
X = X[:, var_mask]
print(f"✅ Retained {X.shape[1]} non-zero-variance features.")

# === Step 3: Subsample for Random Forest importance ===
print("🔍 Subsampling for feature importance training...")
X_sub, _, y_sub, _ = train_test_split(X, labels, train_size=0.2, stratify=labels, random_state=1)

# === Step 4: Train Random Forest ===
print("🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_sub, y_sub)
importances = rf.feature_importances_

'''
# === Step 5: Normalize importances ===
scaler = MinMaxScaler()
normalized_importances = scaler.fit_transform(importances.reshape(-1, 1)).flatten()
print("📊 Normalized importance range:", normalized_importances.min(), "to", normalized_importances.max())
'''

# === Step 5: Normalize importances with safe fallback ===
w = importances.astype(np.float64)
den = w.max() - w.min()
if den < 1e-8:
    normalized_importances = np.ones_like(w)  # neutral scaling if all weights equal
else:
    normalized_importances = (w - w.min()) / den
print("📊 Normalized importance range:", normalized_importances.min(), "to", normalized_importances.max())

# === Step 6: Save feature importances to CSV ===
print("💾 Saving importance values...")
kmer_names = pd.read_csv(args.input, nrows=1).drop(columns=["Label"])
kmer_names = kmer_names.select_dtypes(include=[np.number]).columns[var_mask]
#pd.read_csv(args.input, nrows=1).drop(columns=["Label"]).columns[var_mask]

importance_df = pd.DataFrame({
    "kmer": kmer_names,
    "importance": importances,
    "normalized_importance": normalized_importances
})
importance_df.to_csv(args.importance_out, index=False)
print(f"✅ Feature importance saved to '{args.importance_out}'")

# === Step 7: Apply scaling with NumPy vectorization ===
print("⚙️ Scaling original feature matrix using importances...")
X_scaled = X * normalized_importances  # NumPy broadcasting

# === Step 8: Reattach labels and save scaled dataset ===
df_scaled = pd.DataFrame(X_scaled, columns=kmer_names)
df_scaled["Label"] = labels
df_scaled.to_csv(args.output, index=False)
print(f"✅ Scaled feature matrix saved to '{args.output}'")

# === Step 9: Memory usage (optional) ===
print(f"🧠 Memory used: {psutil.virtual_memory().used / 1e9:.2f} GB")
