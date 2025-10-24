import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Setup Path (Sama kayak P5/P6) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULT_DIR = os.path.join(BASE_DIR, "result")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Set seed acak biar hasilnya konsisten (reproducible)
np.random.seed(42)
tf.random.set_seed(42)

print("--- Memulai Pertemuan 7: Artificial Neural Network (ANN) ---")

# --- Langkah 1: Siapkan Data ---
print("\n[Langkah 1: Siapkan Data]")
df = pd.read_csv(os.path.join(DATASET_DIR, "processed_kelulusan.csv"))
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# PENTING: ANN sangat sensitif sama skala data.
# Kita WAJIB scaling datanya SEBELUM di-split (sesuai instruksi P7)
sc = StandardScaler()
Xs = sc.fit_transform(X)

# Split 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Shape: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# --- Langkah 2: Bangun Model ANN ---
print("\n[Langkah 2: Bangun Arsitektur ANN]")
model = keras.Sequential([
    # Input layer: jumlah fitur = jumlah kolom di X_train
    layers.Input(shape=(X_train.shape[1],)),
    # Hidden layer 1: 32 neuron, aktivasi ReLU
    layers.Dense(32, activation="relu"),
    # Dropout layer: "mematikan" 30% neuron secara acak pas training (biar gak overfitting)
    layers.Dropout(0.3),
    # Hidden layer 2: 16 neuron, aktivasi ReLU
    layers.Dense(16, activation="relu"),
    # Output layer: 1 neuron, aktivasi Sigmoid (karena ini klasifikasi biner 0/1)
    layers.Dense(1, activation="sigmoid")
])

# Compile model: tentukan "wasit" (loss) dan "pelatih" (optimizer)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy","AUC"]) # AUC = metrik bagus buat klasifikasi

model.summary() # Tampilkan arsitektur model

# --- Langkah 3: Training dengan Early Stopping ---
print("\n[Langkah 3: Training Model]")
# Early Stopping: Berhenti latihan kalo performa di data validasi (val_loss) gak membaik
es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# Mulai training...
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val), # Data buat validasi
    epochs=100,                     # Maksimal 100 putaran
    batch_size=32,
    callbacks=[es],                 # Pake Early Stopping
    verbose=1                       # Tampilkan proses training
)

# --- Langkah 4: Evaluasi di Test Set ---
print("\n[Langkah 4: Evaluasi (Test Set)]")
loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")

# Dapetin probabilitas prediksi
y_proba = model.predict(X_test).ravel()
# Ubah probabilitas jadi 0 atau 1 (threshold 0.5)
y_pred = (y_proba >= 0.5).astype(int)

print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred, digits=3))

# --- Langkah 5: Visualisasi Learning Curve ---
print("\n[Langkah 5: Visualisasi Learning Curve]")
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Learning Curve (P7)")

# Simpan plot ke folder 'result'
plot_path = os.path.join(RESULT_DIR, "p7_learning_curve.png")
plt.tight_layout(); plt.savefig(plot_path, dpi=120)
print(f"Plot Learning Curve disimpan ke {plot_path}")

print("\n--- Pertemuan 7 Selesai ---")