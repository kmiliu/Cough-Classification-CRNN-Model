from pathlib import Path
import os
import shutil
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# === 路径设置 ===
BASE_DIR = str(Path(__file__).resolve().parents[1] / 'output')
RAW_TRAIN = os.path.join(BASE_DIR, "processed_audio/train")
RAW_TEST = os.path.join(BASE_DIR, "processed_audio/test")
AUG_TRAIN = os.path.join(BASE_DIR, "processed_audio_augmented/train")
AUG_TEST = os.path.join(BASE_DIR, "processed_audio_augmented/test")
CLEAN_TRAIN = os.path.join(BASE_DIR, "processed_audio_clean/train")
CLEAN_TEST = os.path.join(BASE_DIR, "processed_audio_clean/test")
FEATURES_TRAIN = os.path.join(BASE_DIR, "features_train.csv")
FEATURES_TEST = os.path.join(BASE_DIR, "features_test.csv")

# === Step 1: 删除错误的增强 test 文件夹 ===
if os.path.exists(AUG_TEST):
    shutil.rmtree(AUG_TEST)
    print("🧹 Removed wrongly augmented test folder")

# === Step 2: 合并原始 train + augmented/train ===
os.makedirs(CLEAN_TRAIN, exist_ok=True)
for src in [RAW_TRAIN, AUG_TRAIN]:
    if not os.path.exists(src):
        print(f"⚠️ Source missing: {src}")
        continue
    for path, _, files in os.walk(src):
        for f in files:
            if not f.endswith(".wav"):
                continue
            src_path = os.path.join(path, f)
            rel = os.path.relpath(src_path, src)
            dst_path = os.path.join(CLEAN_TRAIN, rel)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
print("✅ Merged RAW + AUG train sets")

# === Step 3: 复制 test ===
os.makedirs(CLEAN_TEST, exist_ok=True)
for path, _, files in os.walk(RAW_TEST):
    for f in files:
        if not f.endswith(".wav"):
            continue
        src_path = os.path.join(path, f)
        rel = os.path.relpath(src_path, RAW_TEST)
        dst_path = os.path.join(CLEAN_TEST, rel)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
print("✅ Copied test set")

# === Step 4: 特征提取函数 ===
def extract_features(audio_path, sr=22050):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        if len(y) < sr * 0.5:
            return None
        y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        features = {
            **{f"mfcc_{i+1}": np.mean(mfcc[i]) for i in range(13)},
            **{f"chroma_{i+1}": np.mean(chroma[i]) for i in range(12)},
            "spec_centroid": spec_centroid,
            "spec_bw": spec_bw,
            "rolloff": rolloff,
            "zcr": zcr,
        }
        return features
    except Exception as e:
        print(f"⚠️ Error processing {audio_path}: {e}")
        return None

def process_folder(folder, output_csv):
    rows = []
    for path, _, files in os.walk(folder):
        for f in tqdm(files, desc=f"Extracting {os.path.basename(folder)}"):
            if f.endswith(".wav"):
                full_path = os.path.join(path, f)
                feats = extract_features(full_path)
                if feats:
                    feats["filename"] = os.path.relpath(full_path, folder)
                    rows.append(feats)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"💾 Saved {len(df)} feature rows → {output_csv}")

# === Step 5: 提取特征 ===
process_folder(CLEAN_TRAIN, FEATURES_TRAIN)
process_folder(CLEAN_TEST, FEATURES_TEST)

print("🎉 All done! Clean datasets and features ready.")
