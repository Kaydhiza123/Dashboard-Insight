# 🧠 Smart Digital Twin System for Personal Productivity Prediction
**Capstone Project — Data Science & Machine Learning**

Dashboard Interaktif: [Link Dashboard Streamlit Anda Di Sini](https://dashboard-insight-bgexl5aecgppa9kfdxdkfu.streamlit.app/)

---

## 📌 1. Business Understanding & Masalah
Banyak individu maupun karyawan kesulitan mempertahankan konsistensi performa harian karena tidak memahami pola aktivitas longitudinal mereka. Mayoritas aplikasi produktivitas saat ini hanya bersifat pelacak (*tracking*) pasif tanpa kemampuan memberikan analisis prediktif terhadap risiko penurunan performa kognitif.

**Solusi:** Proyek ini mengembangkan *Smart Digital Twin System*—sebuah sistem replika digital personal yang mampu menganalisis riwayat aktivitas harian, menghitung akumulasi indeks kelelahan, serta memprediksi status produktivitas harian guna mencegah risiko *burnout*.

---

## 📊 2. Metodologi Data Wrangling & Rekayasa Fitur
Proyek ini mengolah dataset longitudinal aktivitas harian sebanyak **73.100 baris data** berskala publik. Alur pemrosesan data meliputi:
* **Gathering Data:** Memuat data historis aktivitas pengguna secara end-to-end.
* **Assessing & Cleaning:** Mengatasi duplikasi data (1.096 baris dihapus), konversi tipe temporal, serta penanganan *outlier* menggunakan teknik **IQR Capping (Winsorization)**.
* **Feature Engineering:** Membentuk fitur-fitur baru yang lebih informatif bagi model, seperti:
    * `completion_ratio`: Rasio penyelesaian tugas (`task_completed` / `task_planned`).
    * `fatigue_index`: Indeks kelelahan instan berdasarkan jam kerja, stres, dan waktu tidur.
    * `cumulative_fatigue`: Akumulasi kelelahan jangka panjang menggunakan metode *Exponential Moving Average* (EMA) dengan decay 10%.

---

## 🤖 3. Eksperimen Model AI & A/B Testing
Untuk memprediksi kelas target produktivitas (`productivity_label`), dilakukan eksperimen perbandingan (*A/B Testing*) antara dua arsitektur Machine Learning menggunakan skema *Stratified K-Fold Cross Validation*:

* **Model A (Random Forest Classifier):** Mencapai akurasi sebesar **61.93%**.
* **Model B (Logistic Regression):** Mencapai akurasi sebesar **62.65%**.

**Kesimpulan Statistik:** Model B (Logistic Regression) unggul sebesar **+0.72%** dibandingkan Model A. Melalui uji hipotesis, diperoleh nilai *p-value* sebesar **0.0064** (< 0.05) yang menandakan bahwa keunggulan performa Model B **signifikan secara statistik** dan dipilih sebagai mesin penggerak utama pada sistem Digital Twin ini.

---

## 🖥️ 4. Struktur Repositori Proyek
```text
.
├── app.py                         # File utama aplikasi Dashboard Streamlit
├── final_dataset_model_ready .csv # Dataset final bersih (73.100 baris)
├── requirements.txt               # Daftar library dependensi python
└── README.md                      # Dokumentasi proyek
