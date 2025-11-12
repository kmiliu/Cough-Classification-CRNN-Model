import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# === Paths ===
TRAIN_CSV = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/model_input_clean_train.csv"
TEST_CSV  = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/model_input_clean_test.csv"

# === Load data ===
train = pd.read_csv(TRAIN_CSV, encoding='utf-8')
test  = pd.read_csv(TEST_CSV, encoding='utf-8')

# Keep only numeric columns
train = train.select_dtypes(include=[np.number])
test = test.select_dtypes(include=[np.number])
print("✅ Numeric-only data kept:")
print(train.info())

# === Split features and labels ===
X_train = train.drop(columns=['covid_test_status'])
y_train = train['covid_test_status'].astype(int)
X_test = test.drop(columns=['covid_test_status'])
y_test = test['covid_test_status'].astype(int)

print(f"✅ Loaded train: {X_train.shape}, test: {X_test.shape}")
print("Train label distribution:\n", y_train.value_counts())
print("Test label distribution:\n", y_test.value_counts())

# === Standardize ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Reshape for CNN/GRU ===
X_train_scaled = X_train_scaled[..., np.newaxis]  # (samples, features, 1)
X_test_scaled = X_test_scaled[..., np.newaxis]

# === Class weights ===
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {cls: w for cls, w in zip(classes, weights)}
print("Class weights:", class_weights)

# === Build CRNN model ===
model = Sequential([
    Input(shape=(X_train_scaled.shape[1], 1)),

    Conv1D(64, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    GRU(128, return_sequences=False),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# === Callbacks ===
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
]

# === Train ===
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# === Evaluate ===
y_pred_prob = model.predict(X_test_scaled).ravel()
auc = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
best_thresh = thresholds[np.argmax(tpr - fpr)]

print(f"\n🌟 AUC (ROC): {auc:.4f}")
print(f"🎯 Best threshold = {best_thresh:.3f}")

y_pred = (y_pred_prob >= best_thresh).astype(int)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# === Plot ROC Curve ===
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CRNN ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()

# === Save model ===
os.makedirs("/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/model_results", exist_ok=True)
model.save("/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/model_results/crnn_model.h5")
print("💾 Saved model to output/model_results/crnn_model.h5")
