#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train ML baselines on precomputed features (MFCC/Chroma/Spectral/ZCR).
- Uses model_input_clean_train.csv / model_input_clean_test.csv
- Trains LogisticRegression and RandomForest
- Handles class imbalance (class_weight + threshold tuning)
- Prints metrics + saves predictions
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# ===== Paths =====
ROOT = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset"
OUT_DIR = os.path.join(ROOT, "output", "model_results")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(ROOT, "output", "model_input_clean_train.csv")
TEST_CSV  = os.path.join(ROOT, "output", "model_input_clean_test.csv")

# ===== Load =====
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

label_col = "covid_test_status"
feature_cols = [c for c in train_df.columns if c.startswith(("mfcc_", "chroma_", "spec_", "zcr"))]

X_train = train_df[feature_cols].values
y_train = train_df[label_col].astype(int).values
X_test  = test_df[feature_cols].values
y_test  = test_df[label_col].astype(int).values

print(f"✅ Loaded train: {X_train.shape}, test: {X_test.shape}")
print("Train label distribution:\n", pd.Series(y_train).value_counts())
print("Test  label distribution:\n", pd.Series(y_test).value_counts())

# ===== Scale =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ===== Helper: threshold tuning =====
def evaluate_with_threshold(y_true, y_prob, name: str):
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    # choose threshold maximizing Youden's J
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
    y_pred = (y_prob >= best_thr).astype(int)
    print(f"\n🌟 {name} | AUC={auc:.4f} | Best threshold={best_thr:.3f}")
    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    return auc, best_thr

# ===== Class weights (to reduce imbalance) =====
classes = np.unique(y_train)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(k): float(v) for k, v in zip(classes, cw)}
print("Class weight:", class_weight)

# ===== Model 1: Logistic Regression =====
logreg = LogisticRegression(
    max_iter=5000,
    class_weight=class_weight,
    solver="liblinear",
    C=1.0
)
logreg.fit(X_train_scaled, y_train)
proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]
evaluate_with_threshold(y_test, proba_lr, "LogisticRegression")

# ===== Model 2: Random Forest =====
rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)  # tree models don't need scaling
proba_rf = rf.predict_proba(X_test)[:, 1]
evaluate_with_threshold(y_test, proba_rf, "RandomForest")

# ===== Save predictions =====
pd.DataFrame({
    "y_test": y_test,
    "proba_lr": proba_lr,
    "proba_rf": proba_rf
}).to_csv(os.path.join(OUT_DIR, "baseline_predictions.csv"), index=False)

print(f"\n💾 Saved predictions to {os.path.join(OUT_DIR, 'baseline_predictions.csv')}")
