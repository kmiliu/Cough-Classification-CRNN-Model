import os
import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# === Paths ===
TRAIN_CSV = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/model_input_clean_train.csv"
TEST_CSV = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/model_input_clean_test.csv"
SAVE_MODEL_PATH = "/Users/kemingliu/Desktop/CoronaHack-Respiratory-Sound-Dataset/output/model_results/cnn_baseline.keras"

# === Safe CSV loader ===
def safe_read_csv(path):
    """Try UTF-8 first, fallback to Latin-1."""
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"⚠️ UTF-8 decode failed for {path}, using latin1 instead.")
        with open(path, 'r', encoding='latin1') as f:
            content = f.read()
        return pd.read_csv(io.StringIO(content))

# Load train/test
train = safe_read_csv(TRAIN_CSV)
test = safe_read_csv(TEST_CSV)

# === Clean up non-numeric columns ===
train = train.select_dtypes(include=[np.number])
test = test.select_dtypes(include=[np.number])

train = train.dropna(subset=["covid_test_status"])
test = test.dropna(subset=["covid_test_status"])

train["covid_test_status"] = train["covid_test_status"].astype(int)
test["covid_test_status"] = test["covid_test_status"].astype(int)

print("✅ Numeric-only data kept:")
print(train.info())


# Identify label column
label_col = "covid_test_status"
feature_cols = [c for c in train.columns if c != label_col]

X_train = train[feature_cols].values
y_train = train[label_col].values
X_test = test[feature_cols].values
y_test = test[label_col].values

print(f"✅ Loaded train: {X_train.shape}, test: {X_test.shape}")
print("Train label distribution:\n", pd.Series(y_train).value_counts())
print("Test label distribution:\n", pd.Series(y_test).value_counts())

# === Scale data ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for CNN (samples, timesteps, features=1)
X_train_scaled = np.expand_dims(X_train_scaled, axis=2)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)

# === Compute class weights ===
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# === Build CNN model ===
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Train model ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
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

# === Save model ===
os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
model.save(SAVE_MODEL_PATH)
print(f"\n💾 Model saved to {SAVE_MODEL_PATH}")
