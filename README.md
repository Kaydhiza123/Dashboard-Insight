# 🧠 Smart Digital Twin System for Personal Productivity Prediction
**Capstone Project — Data Science & Machine Learning**

Dashboard Interaktif: [Link Dashboard Streamlit Proyek](https://dashboard-capstone-psu061.streamlit.app/)

---

Proyek ini membahas rancangan dan implementasi **Smart Digital Twin System** untuk memprediksi dan menganalisis produktivitas personal menggunakan pendekatan berbasis Data Science dan Deep Learning. Menggunakan dataset longitudinal *Personal Productivity Prediction* oleh Nesya Enjeliyah, proyek ini mensimulasikan kebiasaan harian dari 100 pengguna selama periode 2 tahun untuk membangun representasi digital (*digital twin*) dari pola produktivitas mereka.

* **Dataset:** *Personal Productivity Prediction* by Nesya Enjeliyah
* **Sumber Data Awal:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/nesyaenjeliyah/personal-productivity-prediction/data)

---

## 📋 Daftar Isi
1. [Business Understanding & Identifikasi Masalah](#1-business-understanding--identifikasi-masalah)
2. [Data Gathering & Dataset Profile](#2-data-gathering--dataset-profile)
3. [Data Assessing & Quality Control](#3-data-assessing--quality-control)
4. [Data Cleaning & Feature Engineering Pipeline](#4-data-cleaning--feature-engineering-pipeline)
5. [Advanced Mathematical Formulations](#5-advanced-mathematical-formulations)
6. [Exploratory Data Analysis (EDA) & Insights](#6-exploratory-data-analysis-eda--insights)
7. [Eksperimen Model AI & A/B Testing](#7-eksperimen-model-ai--ab-testing)
8. [Persiapan Data untuk Analisis Runtun Waktu (LSTM)](#8-persiapan-data-untuk-analisis-runtun-waktu-lstm)
9. [Model-Ready Dataset & Data Dictionary](#9-model-ready-dataset--data-dictionary)
10. [Struktur Repositori Proyek](#10-struktur-repositori-proyek)
11. [Rekomendasi Tahap Selanjutnya](#11-rekomendasi-tahap-selanjutnya)

---

## 🎯 1. Business Understanding & Identifikasi Masalah

### 1.1 Identifikasi Masalah
Banyak individu maupun karyawan mengalami kesulitan dalam menjaga konsistensi produktivitas karena tidak memahami pola aktivitas harian mereka secara mendalam. Mayoritas aplikasi produktivitas saat ini sebagian besar masih berfungsi sebagai alat pencatatan pasif (*tracking tool*) tanpa disertai modul analisis mendalam, prediksi perilaku, atau kemampuan memberikan analisis prediktif terhadap risiko penurunan performa kognitif.

### 1.2 Solusi yang Dikembangkan
Membangun **Smart Digital Twin System for Personal Productivity Prediction**—sebuah sistem replika digital personal berbasis AI. Dengan menganalisis pola perilaku historis, sistem mampu mendeteksi riwayat aktivitas harian, menghitung akumulasi indeks kelelahan, memprediksi status produktivitas harian, serta mencegah risiko *burnout* dengan memberikan rekomendasi waktu aktivitas optimal secara *real-time*.

### 1.3 Business Questions (BQ) & Metodologi Analisis

| No | Business Question | Metode Analisis | Output Sistem / Aplikasi |
|---|---|---|---|
| **BQ1** | Faktor kebiasaan harian apa yang paling berpengaruh terhadap skor produktivitas pengguna? | Korelasi Linear & Feature Importance | Dashboard Insight Utama |
| **BQ2** | Bagaimana pola aktivitas harian pengguna berubah dari waktu ke waktu dan apakah terdapat tren konsisten? | Time Series Analysis (Rolling Average) | Grafik Tren Temporal Kontinu |
| **BQ3** | Apakah tingkat stres dan mood berpengaruh signifikan terhadap jumlah task yang diselesaikan? | Korelasi Korelatif & Scatter Analysis | Insight Card & Alert System |
| **BQ4** | Seberapa efektif pengguna dalam menyelesaikan task yang direncanakan, dan pola apa yang membedakan pengguna efisien? | Task Completion Rate + EDA | Skor Konsistensi & Profiling |
| **BQ5** | Apakah pengguna dengan downtime tinggi dan exercise rendah cenderung memiliki focus score dan mood score lebih rendah? | Scatter Plot & Segmentasi Komparatif | Rekomendasi AI Interaktif |
| **BQ6** | Apakah terdapat pola musiman (mingguan/bulanan) pada produktivitas yang dapat digunakan sebagai dasar prediksi LSTM? | Time Series Decomposition (Observed, Trend, Seasonal, Residual) | Prediksi Pola Masa Depan |

---

## 📊 2. Data Gathering & Dataset Profile

Dataset yang digunakan merupakan data sintetis longitudinal berskala besar yang mensimulasikan aktivitas harian 100 pengguna unik selama periode 2 tahun secara berurutan (*time series data*).
* **Dimensi Awal Data:** 74,196 baris × 15 kolom.
* **Karakteristik Data:** Resolusi harian per pengguna yang merekam durasi tidur, durasi kerja, jumlah tugas selesai, tingkat kebugaran, tingkat stres, hingga indikator psikologis (mood dan tingkat fokus).

---

## 🔍 3. Data Assessing & Quality Control

Sebelum dilakukan pembersihan dan pemodelan, dilakukan audit kualitas data (*data assessment*) secara menyeluruh dengan temuan sebagai berikut:
1. **Missing Values:** Ditemukan sebanyak **31,880 missing values** yang tersebar di beberapa fitur utama seperti `exercise_duration` (5.99%), `sleep_duration` (5.01%), `stress_level` (4.99%), `study_work_duration` (4.99%), dan `focus_score` (4.99%).
2. **Data Duplikat:** Terdapat **1,096 baris data duplikat** yang wajib dieliminasi agar tidak membiaskan model.
3. **Deteksi Outlier (Metode IQR):** Ditemukan sejumlah nilai ekstrem pada fitur `sleep_duration` (537 baris), `downtime_duration` (443 baris), dan `mood_score` (301 baris).
4. **Plausibility Check (Validitas Skala):** Ditemukan anomali nilai di luar rentang fungsional logika fisis, seperti nilai negatif pada durasi kerja/belajar (`study_work_duration` sebanyak 50 baris), nilai negatif pada jumlah task selesai (`task_completed` sebanyak 30 baris), serta nilai di luar skala rentang [1-10] untuk parameter psikologis (`stress_level`, `mood_score`, `focus_score`).

---

## 🛠️ 4. Data Cleaning & Feature Engineering Pipeline

Tahap pembersihan dilakukan melalui sebuah data pipeline yang ketat untuk menjamin *data integrity*:
* **Hapus Duplikat:** 1,096 baris data duplikat dihapus secara permanen, menyisakan **73,100 baris data bersih** skala publik.
* **Imputasi Missing Values:** Karena data ini bersifat runtun waktu (*time series longitudinal*), penanganan *missing values* menggunakan metode **Forward-Fill (ffill)** dan **Backward-Fill (bfill)** yang dikelompokkan berdasarkan `user_id`. Sisa data kosong yang tidak ter-cover diisi menggunakan nilai median dari masing-masing fitur.
* **Winsorization (Capping Outliers):** Nilai outlier ekstrem dibatasi (*capped*) menggunakan ambang batas atas dan batas bawah dari metode Interquartile Range (IQR) agar distribusi data kembali normal tanpa kehilangan baris data berharga.
* **Transformasi Kolom:** Fitur `workday` diubah namanya menjadi `is_weekend` dengan membalik nilai booleannya (0 untuk hari kerja, 1 untuk akhir pekan) guna menyelaraskan skema data dengan arsitektur database operasional (Prisma/PostgreSQL).
* **Sanity Checks & Hard Clipping:** Dilakukan pengecekan akhir dan pemaksaan batas (*hard clipping*) untuk memastikan parameter durasi berada dalam rentang fisis [0-24 jam] atau [0-100%] dan parameter skala psikologis tepat berada di rentang [1-10].

---

## 🧮 5. Advanced Mathematical Formulations

Untuk membangun representasi *Digital Twin* yang akurat, diimplementasikan beberapa rekayasa fitur tingkat lanjut menggunakan landasan matematis dan asumsi akademis yang kuat:

### 5.1 Break Duration
Durasi istirahat non-produktif namun menyegarkan dihitung berdasarkan sisa siklus waktu aktif harian seseorang:
$$\text{Break Duration} = 24 - \left(\text{Sleep Duration} + \text{Study Work Duration} + \text{Downtime Duration} + \frac{\text{Exercise Duration}}{60}\right)$$

*Nilai dipotong secara logis pada rentang [0, 6] jam.*

### 5.2 Task Planned & Completion Ratio
Jumlah tugas yang direncanakan disintesis secara proporsional terhadap durasi kerja pengguna:
$$\text{Task Planned} = \text{clip}\left(\text{round}\left(\text{Study Work Duration} \times 1.2\right), 3, 15\right)$$
$$\text{Completion Ratio} = \frac{\text{Task Completed}}{\text{Task Planned}}$$

### 5.3 Productivity Score
Skor produktivitas komprehensif $[0-100]$ dirancang menggunakan teknik pembobotan multi-kriteria (*multi-criteria weighted index*) berdasarkan kontribusi masing-masing aktivitas harian:
$$\text{Productivity Score} = \left( 0.341 \times \text{Completion Ratio} + 0.224 \times \frac{\text{Focus Score}}{10} + 0.152 \times \frac{\text{Break Duration}}{6} + 0.067 \times \frac{\text{Sleep Duration}}{\text{Sleep Max}} + 0.064 \times \frac{10 - \text{Stress Level}}{10} + 0.063 \times \left(1 - \frac{\text{Downtime Duration}}{24}\right) + 0.054 \times \frac{\text{Exercise Duration}}{\text{Exercise Max}} + 0.034 \times \frac{\text{Mood Score}}{10} \right) \times 100$$
*Skor kemudian dihaluskan (*smoothed*) menggunakan teknik **3-day rolling mean** per pengguna untuk meminimalkan fluktuasi harian acak dan menangkap tren produktivitas riil.*

### 5.4 Fatigue Index & Cumulative Fatigue
Indeks kelelahan harian dirancang untuk memantau beban fisik dan mental pengguna:
$$\text{Fatigue Index} = 40 \times \left(\frac{\text{Stress Level}}{\text{Stress Max}}\right) + 30 \times \left(\frac{\text{Downtime Duration}}{\text{Downtime Max}}\right) + 30 \times \left(\frac{\text{Study Work Duration}}{\text{Study Max}}\right)$$

Untuk menghitung akumulasi kelelahan multi-hari, diterapkan metode **Exponential Moving Average (EMA)** dengan koefisien peluruhan (*decay factors*) pemulihan sebesar 10% per hari:
$$\text{Cumulative Fatigue}(t) = 0.10 \times \text{Fatigue Index}(t) + 0.90 \times \text{Cumulative Fatigue}(t-1)$$

### 5.5 Validasi Hubungan Fitur
Berdasarkan uji statistik korelasi Pearson, ditemukan **korelasi negatif yang signifikan sebesar -0.3926** antara `productivity_score` dan `fatigue_index` dengan *p-value* = 0.0000. Hal ini memvalidasi secara ilmiah logika sistem: **semakin tinggi tingkat kelelahan akumulatif seseorang, maka performa produktivitasnya akan menurun secara signifikan**.

---

## 📈 6. Exploratory Data Analysis (EDA) & Insights

* **Dampak Jam Kerja Berlebih:** Fitur yang memiliki korelasi linear paling kuat terhadap `productivity_score` adalah `study_work_duration` (-0.46) dan `break_duration` (+0.44). Durasi kerja yang terlalu panjang terbukti bersifat **kontraproduktif** (menurunkan skor produktivitas) karena memicu kejenuhan, sedangkan alokasi waktu istirahat (`break_duration`) menjadi pendorong utama kestabilan performa.
* **Pola Produktivitas Mingguan (BQ6):** Tren performa puncak (*peak performance*) pengguna rata-rata tercapai pada **pertengahan minggu (Rabu dan Kamis)**, sedangkan tingkat fokus dan produktivitas merosot tajam pada akhir pekan (Sabtu dan Minggu) seiring peningkatan durasi waktu santai (*downtime*).

---

## 🤖 7. Eksperimen Model AI & A/B Testing

Untuk memprediksi kelas target produktivitas (`productivity_label`), dilakukan eksperimen perbandingan (*A/B Testing*) mendalam antara dua arsitektur Machine Learning menggunakan skema evaluasi *Stratified K-Fold Cross Validation*:

* **Model A (Random Forest Classifier):** Mencapai tingkat akurasi baseline sebesar **61.93%**. Model ini efektif dalam membedakan kelas mayoritas, namun membutuhkan optimasi pada batas ambang kelas kritis `At Risk`.
* **Model B (Logistic Regression):** Mencapai tingkat akurasi sebesar **62.65%**.

**Kesimpulan Statistik:** Model B (Logistic Regression) unggul sebesar **+0.72%** dibandingkan Model A. Melalui uji hipotesis, diperoleh nilai *p-value* sebesar **0.0064** (< 0.05). Hal ini menandakan bahwa keunggulan performa Model B **signifikan secara statistik** dan dipilih sebagai mesin penggerak utama pada sistem Digital Twin ini untuk melampaui performa generalisasi klasifikasi klasik.

---

## ⏱️ 8. Persiapan Data untuk Analisis Runtun Waktu (LSTM)

Mengingat dataset ini memiliki karakteristik *sequential time series* yang kuat dengan **lag correlation mencapai 0.7168**, data dipersiapkan agar kompatibel dengan arsitektur **Long Short-Term Memory (LSTM)**:
1. Data diurutkan secara ketat kronologis berdasarkan `user_id` dan `date`.
2. Dilakukan pengecekan kestabilan jarak temporal (memastikan tidak ada tanggal yang melompat/terlewat pada setiap pengguna).
3. Nilai target diskrit dipetakan menjadi `today` (label kelas target):
   * **0 (`At Risk`)**: `productivity_score` < 55 (Beresiko *burnout*/penurunan performa)
   * **1 (`Steady`)**: 55 $\le$ `productivity_score` $<$ 70 (Stabil dan konsisten)
   * **2 (`Thriving`)**: `productivity_score` $\ge$ 70 (Sangat produktif dan berkembang)

---

## 📖 9. Model-Ready Dataset & Data Dictionary

Berikut adalah referensi kamus data skema akhir dari dataset yang siap dimasukkan ke dalam model pembelajaran mesin (*Model-Ready Dataset*):

| No | Nama Kolom | Tipe Data | Peran Fitur | Satuan | Rentang Nilai | Keterangan |
|---|---|---|---|---|---|---|
| 1 | `user_id` | INT | Identifier | - | - | ID unik pelacak setiap pengguna |
| 2 | `date` | DATE | Temporal | - | - | Tanggal pencatatan aktivitas harian |
| 3 | `is_weekend` | INT | Temporal | Biner | 0 atau 1 | Indikator hari: 0 = Hari Kerja, 1 = Akhir Pekan |
| 4 | `sleep_duration` | FLOAT | Feature | Jam | 4.71 – 9.71 | Durasi tidur malam pengguna |
| 5 | `study_work_duration`| FLOAT | Feature | Jam | 0.00 – 17.41 | Total durasi pengerjaan tugas atau kerja |
| 6 | `task_completed` | INT | Feature | Count | 0 – 20 | Jumlah tugas yang berhasil diselesaikan |
| 7 | `exercise_duration` | FLOAT | Feature | Menit | 0.00 – 84.50 | Durasi olahraga/aktivitas fisik pengguna |
| 8 | `downtime_duration` | FLOAT | Feature | Jam | 1.85 – 11.21 | Waktu santai non-produktif (screen time) |
| 9 | `stress_level` | INT | Feature | Skala | 1 – 10 | Nilai tingkat stres harian (Max: 8.0) |
| 10 | `mood_score` | INT | Feature | Skala | 1 – 10 | Indikator suasana hati pengguna |
| 11 | `focus_score` | INT | Feature | Skala | 1 – 10 | Indikator tingkat kefokusan mental |
| 12 | `break_duration` | FLOAT | Feature | Jam | 0.00 – 6.00 | [FE] Estimasi waktu istirahat harian |
| 13 | `task_planned` | INT | Konteks | Count | 3 – 15 | [FE] Target jumlah tugas yang direncanakan |
| 14 | `completion_ratio` | FLOAT | Derived | Rasio | 0.00 – 1.00 | [FE] Rasio penyelesaian tugas (`task_completed` / `task_planned`) |
| 15 | `fatigue_index` | FLOAT | Feature | Skor | 12.64 – 89.75 | [FE] Nilai indeks kelelahan harian instan |
| 16 | `cumulative_fatigue` | FLOAT | Feature | Skor | 35.15 – 76.82 | [FE] Akumulasi kelelahan jangka panjang (EMA) |
| 17 | `productivity_score` | FLOAT | Target | Skor | 25.46 – 89.52 | [FE] Nilai akhir performa produktivitas (3-day rolling) |
| 18 | `today` | INT | Target Label| Kategori| 0, 1, atau 2 | [FE] Kelas status: 0=At Risk, 1=Steady, 2=Thriving |

---

## 🖥️ 10. Struktur Repositori Proyek

```text
.
├── app.py                         # File utama aplikasi Dashboard Streamlit
├── final_dataset_model_ready.csv  # Dataset final bersih dan rekayasa fitur (73.100 baris)
├── requirements.txt               # Daftar library dependensi Python
└── README.md                      # Dokumentasi komprehensif proyek
