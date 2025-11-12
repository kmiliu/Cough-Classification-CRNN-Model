import pandas as pd
import os

# === 路径配置 ===
BASE_DIR = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output"
FEATURES_TRAIN = os.path.join(BASE_DIR, "features_train.csv")
FEATURES_TEST = os.path.join(BASE_DIR, "features_test.csv")
META_CSV = os.path.join("/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/Corona-Hack-Respiratory-Sound-Metadata.csv")

OUTPUT_TRAIN = os.path.join(BASE_DIR, "model_input_clean_train.csv")
OUTPUT_TEST = os.path.join(BASE_DIR, "model_input_clean_test.csv")

# === Step 1: 读取特征 ===
ftrain = pd.read_csv(FEATURES_TRAIN)
ftest = pd.read_csv(FEATURES_TEST)
print(f"✅ Loaded features: train={len(ftrain)}, test={len(ftest)}")

# === Step 2: 读取 metadata ===
meta = pd.read_csv(META_CSV, encoding='latin1')
print(f"✅ Loaded metadata: {len(meta)} rows")

# 确保 USER_ID 一致性
meta.columns = [c.lower() for c in meta.columns]
meta = meta.rename(columns={"user_id": "userid"})
if "covid_test_status" not in meta.columns:
    raise ValueError("❌ 'covid_test_status' not found in metadata")

meta = meta[meta["covid_test_status"].isin([0, 1])]
meta["covid_test_status"] = meta["covid_test_status"].astype(int)

# === Step 3: 从路径中提取 user_id ===
def extract_user_id(path):
    parts = path.split(os.sep)
    for p in parts:
        if len(p) == 28:  # Coswara 风格的 28 位加密 ID
            return p
    return None

for df in [ftrain, ftest]:
    df["userid"] = df["filename"].apply(extract_user_id)

# === Step 4: 合并 ===
merged_train = pd.merge(ftrain, meta, how="left", on="userid")
merged_test = pd.merge(ftest, meta, how="left", on="userid")

print(f"🧩 After merge: train={len(merged_train)}, test={len(merged_test)}")

# === Step 5: 清理并保存 ===
merged_train = merged_train.dropna(subset=["covid_test_status"])
merged_test = merged_test.dropna(subset=["covid_test_status"])

merged_train.to_csv(OUTPUT_TRAIN, index=False)
merged_test.to_csv(OUTPUT_TEST, index=False)

print("💾 Saved model input files:")
print(f"   → {OUTPUT_TRAIN} ({len(merged_train)} rows)")
print(f"   → {OUTPUT_TEST} ({len(merged_test)} rows)")

# === Step 6: 简要统计 ===
print("\n📊 Label distribution (train):")
print(merged_train["covid_test_status"].value_counts())
print("\n📊 Label distribution (test):")
print(merged_test["covid_test_status"].value_counts())
