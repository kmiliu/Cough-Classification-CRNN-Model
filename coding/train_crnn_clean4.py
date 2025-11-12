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
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight

# === Paths ===
TRAIN_CSV = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/model_input_clean_train.csv"
TEST_CSV  = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/model_input_clean_test.csv"
SAVE_DIR = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/crnn_model_results_4"
os.makedirs(SAVE_DIR, exist_ok=True)

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

# === Oversampling ===
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train_scaled.reshape(len(X_train_scaled), -1), y_train)
X_train_bal = X_train_bal.reshape(X_train_bal.shape[0], X_train_scaled.shape[1], 1)
print(f"🔁 Oversampled training shape: {X_train_bal.shape}, label distribution: {np.bincount(y_train_bal)}")

# === Define Focal Loss ===
def focal_loss(gamma=2., alpha=0.75):
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
              loss=focal_loss(gamma=2.0, alpha=0.75),
              metrics=['accuracy'])
model.summary()

# === Callbacks ===
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(SAVE_DIR, "best_crnn_focal_biGRU.keras"),
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
best_thresh = thresholds_pr[np.argmax(f1_scores)]
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

# === Plot ROC Curve ===
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_roc:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Focal + BiGRU)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "roc_curve_crnn_focal_biGRU.png"))
plt.show()

# === Plot PR Curve ===
plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Focal + BiGRU)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "pr_curve_crnn_focal_biGRU.png"))
plt.show()

# === Plot Training Curves ===
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves_crnn_focal_biGRU.png"))
plt.show()

print(f"💾 Saved model, plots, and reports to: {SAVE_DIR}")
