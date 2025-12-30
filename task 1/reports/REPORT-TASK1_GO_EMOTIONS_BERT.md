# Laporan Task 1: Fine-Tuning BERT untuk NLI (MNLI Dataset)

**Model:** BERT Base Uncased  
**Dataset:** GLUE Benchmark - MNLI  
**Task:** Natural Language Inference (Sequence Classification)

---

## 1. Pendahuluan
Laporan ini mendokumentasikan proses fine-tuning model BERT untuk menyelesaikan tugas **Natural Language Inference (NLI)**. Tugas ini bertujuan menentukan hubungan logika antara dua kalimat (*Premise* dan *Hypothesis*), apakah hubungan tersebut berupa:
1. **Entailment** (Mendukung/Sesuai)
2. **Neutral** (Netral/Tidak Berhubungan)
3. **Contradiction** (Bertentangan)

## 2. Metodologi

### A. Dataset
- **Sumber:** `nyu-mll/glue` (subset `mnli`).
- **Penanganan Dataset (Isu Label -1):** Ditemukan bahwa subset `test_matched` pada dataset GLUE memiliki label tersembunyi (`-1`) yang menyebabkan error pada GPU (*CUDA device-side assert*). Oleh karena itu, data validasi dan pengujian dialihkan menggunakan subset `validation_matched` yang memiliki label valid (0, 1, 2).

### B. Preprocessing
Model BERT menerima input pasangan kalimat yang digabungkan dengan token separator khusus `[SEP]`.
- **Input Format:** `[CLS] Premise [SEP] Hypothesis [SEP]`
- **Tokenisasi:** Truncation dan padding ke `max_length=128`.

### C. Konfigurasi Training
- **Model:** `bert-base-uncased` dengan `num_labels=3`.
- **Hyperparameter:**
  - Epochs: 3
  - Batch Size: 8
  - Learning Rate: 2e-5
- **Metrics:** Accuracy dan F1-Score (Weighted Average).

## 3. Hasil Implementasi

### Visualisasi Training
Proses training berjalan stabil selama 3 epoch. Grafik loss menunjukkan konvergensi yang baik tanpa tanda-tanda overfitting yang parah.

![visual-training-history-goemotions](visual-training-history-goemotions.png)

### Metrik Evaluasi
Karena dataset ini adalah *Multi-Class Classification* (3 kelas), penggunaan parameter `average='weighted'` pada F1 Score diterapkan untuk mendapatkan gambaran performa yang akurat dibandingkan hanya menggunakan akurasi biasa.

## 4. Analisis Inference
Pengujian model dilakukan secara manual dengan kalimat input buatan sendiri.

**Contoh Input:**
- **Premise:** "This is an incredible movie with outstanding performances!"
- **Hypothesis:** "The film is terrible."

**Hasil Prediksi:**
- **Label:** Contradiction
- **Confidence:** Tinggi (>90%)

Hal ini menunjukkan model mampu memahami negasi dan antonim (incredible vs terrible) dalam konteks kalimat.

## 5. Kesimpulan
Tantangan utama dalam pengerjaan tugas ini adalah ketidakcocokan label pada dataset testing bawaan Hugging Face. Setelah mengganti dataset evaluasi ke `validation_matched` dan melakukan hard-reset pada environment GPU, model BERT berhasil dilatih dengan performa yang memuaskan untuk tugas NLI.