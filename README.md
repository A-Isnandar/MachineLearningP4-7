# Tugas Machine Learning (Pertemuan 4-7)

Repositori ini berisi rangkaian tugas mata kuliah Machine Learning (Pertemuan 4-7), yang mencakup alur kerja *end-to-end* mulai dari persiapan data hingga implementasi model *deep learning*.

---

## 📂 Struktur Project

Proyek ini disusun dengan struktur folder standar untuk memisahkan data, kode, dan hasil:

```text
MachineLearningP4-7/
├── dataset/
│   ├── kelulusan_mahasiswa.csv  (Data mentah)
│   └── processed_kelulusan.csv  (Data bersih hasil P4)
├── model/
│   ├── model.pkl                (Model P5: LogReg/RF)
│   └── rf_model.pkl             (Model P6: Random Forest)
├── result/
│   ├── p6_pr_test.png           (Plot PR Curve P6)
│   ├── p6_roc_test.png          (Plot ROC Curve P6)
│   ├── p7_learning_curve.png    (Plot Learning Curve P7)
│   └── roc_test.png             (Plot ROC Curve P5)
├── src/
│   ├── pertemuan_4.py           (Script Data Preparation)
│   ├── pertemuan_5.py           (Script Modeling & API)
│   ├── pertemuan_6.py           (Script Deep Dive Random Forest)
│   └── pertemuan_7.py           (Script Artificial Neural Network)
└── .gitignore                   (File konfigurasi Git)
```

## Library Utama

  * **Data Processing**: `pandas`, `numpy`
  * **Modeling (ML)**: `scikit-learn`
  * **Modeling (DL)**: `tensorflow` (Keras)
  * **Visualisasi**: `matplotlib`, `seaborn`
  * **API (Opsional)**: `flask`
  * **Serialisasi Model**: `joblib`

-----

## Alur Kerja & Penjelasan 

Setiap skrip dalam folder `src/` mewakili satu pertemuan dan saling berkelanjutan.

### 1\. `src/pertemuan_4.py` — Data Preparation

Skrip ini adalah fondasi dari proyek. Tujuannya adalah membersihkan data mentah dan menyiapkannya untuk *modeling*.

  * **Proses**: Membaca `dataset/kelulusan_mahasiswa.csv`, melakukan *cleaning* (cek duplikat, *missing values*), *Exploratory Data Analysis* (EDA) sederhana, dan *feature engineering* (membuat kolom `Rasio_Absensi` dan `IPK_x_Study`).
  * **Output**: Menghasilkan file `dataset/processed_kelulusan.csv` yang bersih dan siap pakai.

### 2\. `src/pertemuan_5.py` — Modeling & API

Skrip ini berfokus pada alur kerja *modeling* standar, membandingkan *baseline* dengan model alternatif, dan (opsional) men-deploy model sebagai API.

  * **Proses**: Membangun *pipeline* `scikit-learn` untuk *preprocessing*. Melatih model *baseline* (**Logistic Regression**) dan model alternatif (**Random Forest**). Melakukan *hyperparameter tuning* sederhana menggunakan `GridSearchCV` untuk menemukan model terbaik.
  * **Output**: Menyimpan model terbaik ke `model/model.pkl` dan menjalankan server API sederhana menggunakan **Flask** pada *endpoint* `/predict`.

### 3\. `src/pertemuan_6.py` — Deep Dive: Random Forest

Skrip ini berfokus khusus untuk mengeksplorasi **Random Forest** secara lebih mendalam.

  * **Proses**: Membangun *pipeline* RF, melakukan **Cross-Validation** (K-Fold) untuk mendapatkan evaluasi yang lebih stabil, dan *tuning* menggunakan `GridSearchCV`.
  * **Output**: Menghasilkan evaluasi mendalam, termasuk **ROC Curve** (`p6_roc_test.png`), **Precision-Recall Curve** (`p6_pr_test.png`), dan analisis **Feature Importance** (fitur apa yang paling berpengaruh). Model final disimpan ke `model/rf_model.pkl`.

### 4\. `src/pertemuan_7.py` — Artificial Neural Network (ANN)

Skrip ini adalah pengenalan ke *deep learning* untuk masalah klasifikasi tabular.

  * **Proses**: Menggunakan **TensorFlow (Keras)** untuk membangun arsitektur *network* sederhana (Sequential model dengan layer `Dense` dan `Dropout`). Data di-scaling menggunakan `StandardScaler` (wajib untuk ANN). Model dilatih menggunakan *callback* **Early Stopping** untuk mencegah *overfitting*.
  * **Output**: Menghasilkan evaluasi model (AUC, F1-Score) dan plot **Learning Curve** (`p7_learning_curve.png`) untuk memvisualisasikan performa *training* vs *validation*.

-----

## Cara Menjalankan

1.  **Clone repositori ini:**

    ```bash
    git clone [https://github.com/A-Isnandar/MachineLearningP4-7.git](https://github.com/A-Isnandar/MachineLearningP4-7.git)
    cd MachineLearningP4-7
    ```

2.  **(Opsional tapi direkomendasikan) Buat dan aktifkan *virtual environment*:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # atau .venv\Scripts\activate di Windows
    ```

3.  **Install *library* yang dibutuhkan:**

    ```bash
    pip install pandas scikit-learn matplotlib seaborn tensorflow joblib flask
    ```

4.  **Jalankan skrip secara berurutan:**

      * Pertama, jalankan P4 untuk membuat data olahan:

        ```bash
        python src/pertemuan_4.py
        ```

      * Kemudian, jalankan skrip pertemuan lainnya:

        ```bash
        python src/pertemuan_5.py
        python src/pertemuan_6.py
        python src/pertemuan_7.py
        ```

-----

## Author

  * **Muhamad Ario Isnandar** - [A-Isnandar](https://github.com/A-Isnandar)

<!-- end list -->