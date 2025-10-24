import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt # Kita butuh ini buat nampilin plot
from sklearn.model_selection import train_test_split

print("--- Memulai Proses Data Preparation ---")

# Langkah 2 — Collection
print("\n[Langkah 2: Collection]")
df = pd.read_csv("dataset/kelulusan_mahasiswa.csv")
print("Info dataset:")
df.info()
print("\nData 5 baris pertama:")
print(df.head())

# Langkah 3 — Cleaning
print("\n[Langkah 3: Cleaning]")
print("Cek missing values:")
print(df.isnull().sum())
print("Menghapus data duplikat...")
df = df.drop_duplicates()
print("Data duplikat sudah dihapus.")

# Visualisasi Boxplot
print("Menampilkan boxplot IPK untuk cek outlier...")
sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK")
plt.show() # Perintah untuk menampilkan plot

# Langkah 4 — Exploratory Data Analysis (EDA)
print("\n[Langkah 4: EDA]")
print("Statistik Deskriptif:")
print(df.describe())

# Visualisasi Histogram
print("Menampilkan histogram IPK...")
sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi IPK")
plt.show()

# Visualisasi Scatterplot
print("Menampilkan scatterplot IPK vs Waktu Belajar...")
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title("IPK vs Waktu Belajar (hue: Lulus)")
plt.show()

# Visualisasi Heatmap Korelasi
print("Menampilkan heatmap korelasi...")
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.show()

# Langkah 5 — Feature Engineering
print("\n[Langkah 5: Feature Engineering]")
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
print("Fitur baru 'Rasio_Absensi' dan 'IPK_x_Study' telah dibuat.")

# Simpan ke CSV baru
df.to_csv("dataset/processed_kelulusan.csv", index=False)
print("Data yang sudah diproses disimpan ke 'processed_kelulusan.csv'")

# Langkah 6 — Splitting Dataset
print("\n[Langkah 6: Splitting Dataset]")
X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Split pertama: 70% Train, 30% Temp (Val + Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=101) # Gua ganti random_state jadi 101

# Split kedua: 50% dari Temp jadi Val, 50% jadi Test (jadi 15% Val, 15% Test)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=101) # Samain random_state

print("Bentuk data (shape):")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

print("\n--- Proses Selesai ---")