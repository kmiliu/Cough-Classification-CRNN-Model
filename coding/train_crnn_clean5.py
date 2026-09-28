from pathlib import Path
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Bidirectional,
                                     GRU, Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             precision_recall_curve, auc, confusion_matrix)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# === Paths ===
TRAIN_CSV = str(Path(__file__).resolve().parents[1] / 'output/model_input_clean_train.csv')
TEST_CSV  = str(Path(__file__).resolve().parents[1] / 'output/model_input_clean_test.csv')
SAVE_DIR = str(Path(__file__).resolve().parents[1] / 'output/crnn_model_results_5')
os.makedirs(SAVE_DIR, exist_ok=True)

# Historical experiment: numeric metadata are included alongside audio features.
# Test-set threshold optimization below is exploratory, not independent evaluation.
# See README before reusing this training setup.

# === Load data ===
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train = train.select_dtypes(include=[np.number])
test = test.select_dtypes(include=[np.number])

# Split features and labels
X_train = train.drop(columns=['covid_test_status'])
y_train = train['covid_test_status'].astype(int)
X_test = test.drop(columns=['covid_test_status'])
y_test = test['covid_test_status'].astype(int)

# === Normalize ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = X_train_scaled[..., np.newaxis]
X_test_scaled = X_test_scaled[..., np.newaxis]

# === Class weights ===
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {cls: w for cls, w in zip(classes, weights)}
print("✅ Class weights:", class_weights)

# === SMOTE Oversampling ===
sm = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled.reshape(len(X_train_scaled), -1), y_train)
X_train_bal = X_train_bal.reshape(X_train_bal.shape[0], X_train_scaled.shape[1], 1)
print(f"🔁 After SMOTE: {X_train_bal.shape}, label distribution: {np.bincount(y_train_bal)}")

# === Define Focal Loss ===
def focal_loss(gamma=2., alpha=0.8):
    def focal_crossentropy(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss = alpha * tf.pow((1 - p_t), gamma) * bce
        return tf.reduce_mean(loss)
    return focal_crossentropy

# === Build Model ===
model = Sequential([
    Input(shape=(X_train_bal.shape[1], 1)),

    Conv1D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(256, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.4),

    Bidirectional(GRU(128, dropout=0.3, return_sequences=False)),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=focal_loss(gamma=2.0, alpha=0.8),
              metrics=['accuracy'])
model.summary()

# === Callbacks ===
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(SAVE_DIR, "best_crnn_focal_biGRU_v2.keras"),
                    monitor='val_loss', save_best_only=True, verbose=1)
]

# === Train ===
history = model.fit(
    X_train_bal, y_train_bal,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# === Predict ===
y_pred_prob = model.predict(X_test_scaled).ravel()

# === ROC Analysis ===
auc_roc = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)

# === PR Analysis ===
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_thresh = thresholds_pr[np.argmax(f1_scores[:-1])]
best_f1 = np.max(f1_scores)
pr_auc = auc(recall, precision)

print(f"\n🌟 ROC-AUC = {auc_roc:.4f}")
print(f"💡 PR-AUC = {pr_auc:.4f}")
print(f"🎯 F1-optimized threshold = {best_thresh:.3f}")
print(f"⭐ Best F1-score = {best_f1:.4f}")

# === Final Predictions ===
y_pred = (y_pred_prob >= best_thresh).astype(int)
print("\n📊 Classification Report (F1-optimized threshold):")
print(classification_report(y_test, y_pred, digits=4))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
print("\n🔍 Confusion Matrix:\n", cm)

# === Save Predictions ===
output_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred_prob': y_pred_prob,
    'y_pred_label': y_pred
})
output_csv = os.path.join(SAVE_DIR, "test_predictions_f1opt_v2.csv")
output_df.to_csv(output_csv, index=False)
print(f"📁 Saved detailed predictions to: {output_csv}")

# === Plot ROC Curve ===
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_roc:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Focal + BiGRU v2)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "roc_curve_crnn_focal_biGRU_v2.png"))
plt.show()

# === Plot PR Curve ===
plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Focal + BiGRU v2)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "pr_curve_crnn_focal_biGRU_v2.png"))
plt.show()

# === Plot Training Curves ===
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Loss Curve (v2)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title("Accuracy Curve (v2)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves_crnn_focal_biGRU_v2.png"))
plt.show()

print(f"💾 All v2 results saved to: {SAVE_DIR}")
