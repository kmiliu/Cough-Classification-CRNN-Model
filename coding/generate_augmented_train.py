import pandas as pd
import os

# Paths
TRAIN_CSV = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/model_input.csv"
AUGMENT_LOG = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/features_augmented.csv"
OUTPUT_CSV = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/model_input_augmented.csv"

# Load original train data
train_df = pd.read_csv(TRAIN_CSV)
print(f"✅ Loaded original train set: {len(train_df)} samples")

# Load augmentation log
aug_df = pd.read_csv(AUGMENT_LOG)
print(f"✅ Loaded augmented samples: {len(aug_df)} samples")

# Align feature columns if necessary
common_cols = [c for c in train_df.columns if c in aug_df.columns]
aug_df = aug_df[common_cols]
train_df = train_df[common_cols]

# Merge
merged_df = pd.concat([train_df, aug_df], ignore_index=True)
print(f"💾 Combined dataset: {len(merged_df)} total samples")

# Label distribution
print("Label distribution:")
print(merged_df["covid_test_status"].value_counts())

# Save
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
merged_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved augmented training set → {OUTPUT_CSV}")
