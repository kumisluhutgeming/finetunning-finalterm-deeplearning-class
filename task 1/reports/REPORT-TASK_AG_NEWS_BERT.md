# Laporan Task 1: Fine-Tuning BERT untuk Klasifikasi Berita (AG News)

**Repository Name:** `finetuning-bert-text-classification`  
**Model:** BERT Base Uncased  
**Dataset:** AG News  
**Task:** Single-Label Text Classification  

---

## 1. Pendahuluan
Laporan ini mendokumentasikan implementasi fine-tuning model **BERT (Bidirectional Encoder Representations from Transformers)** untuk menyelesaikan tugas klasifikasi topik berita. Dataset yang digunakan adalah **AG News**, sebuah dataset benchmark standar yang berisi artikel berita berita pendek.

Tujuan utama dari tugas ini adalah melatih model agar mampu mengkategorikan input teks ke dalam salah satu dari empat kategori topik utama: **World, Sports, Business, atau Sci/Tech**. Berbeda dengan tugas GoEmotions (Multi-Label), tugas ini bersifat **Single-Label Classification**, di mana setiap artikel hanya memiliki tepat satu kategori yang benar.

## 2. Metodologi

### A. Dataset
- **Sumber:** `ag_news` dari Hugging Face Hub.
- **Struktur Label:** Dataset ini memiliki 4 kelas label yang direpresentasikan dalam bentuk integer:
  0. World
  1. Sports
  2. Business
  3. Sci/Tech
- **Distribusi Data:** Dataset terdiri dari 120.000 data latih (train) dan 7.600 data uji (test).

### B. Preprocessing
Proses penyiapan data dilakukan dengan langkah-langkah berikut:
- **Tokenisasi:** Menggunakan `AutoTokenizer` dari `bert-base-uncased` dengan pemotongan (truncation) dan padding hingga `max_length=128`.
- **Penanganan Label:** Karena tugas ini adalah *Single-Label*, kolom label dibiarkan dalam format aslinya (**Integer/Long**). Ini berbeda dengan *Multi-Label* yang membutuhkan konversi ke Float/One-Hot Encoding. Kolom `label` (singular) dari dataset asli diubah namanya menjadi `labels` (plural) agar kompatibel dengan Trainer Hugging Face.

### C. Konfigurasi Model
- **Arsitektur:** `BertForSequenceClassification`.
- **Konfigurasi Spesifik:**
  - `num_labels=4`: Sesuai jumlah kategori berita.
  - `problem_type="single_label_classification"`: Pengaturan ini memastikan model menggunakan *Cross Entropy Loss* yang sesuai untuk klasifikasi tunggal.
- **Hyperparameter Training:**
  - Epochs: 3
  - Batch Size: 8
  - Learning Rate: 2e-5

## 3. Hasil Implementasi

### Visualisasi Training
Proses training berjalan selama 3 epoch. Grafik *Training Loss* menunjukkan penurunan yang konsisten, menandakan model berhasil meminimalkan kesalahan prediksi seiring berjalannya waktu.

![traininglossagnews](traininglossagnews.png)

### Metrik Evaluasi
Evaluasi model dilakukan menggunakan dua metrik utama:
1. **Accuracy:** Untuk mengukur persentase prediksi yang tepat secara keseluruhan.
2. **F1-Score (Weighted):** Memberikan rata-rata harmonik antara presisi dan recall, yang dihitung secara proporsional terhadap jumlah sampel tiap kelas.

## 4. Analisis Inference
Pengujian model (Inference) dilakukan menggunakan pendekatan **Softmax** dan **Argmax**. Berbeda dengan *Multi-Label* yang menggunakan threshold, pada kasus ini model memilih satu kelas dengan probabilitas tertinggi.

**Contoh Input:**
> *"Apple just announced the new iPhone with a revolutionary AI chip that changes everything."*

**Proses Prediksi:**
1. Model menghasilkan *logits* untuk 4 kelas.
2. Fungsi Softmax mengubah logits menjadi probabilitas (total 100%).
3. Fungsi Argmax mengambil indeks dengan nilai tertinggi.

**Hasil Prediksi:**
- **Kategori Terprediksi:** `Sci/Tech`
- **Confidence:** Tinggi (Sangat yakin karena kata kunci seperti "Apple", "iPhone", "AI chip" sangat dominan di kategori teknologi).

## 5. Kesimpulan
Implementasi Fine-Tuning BERT pada dataset AG News berjalan sukses. Perbedaan mendasar dalam penanganan tipe data label (Integer vs Float) dan pemilihan fungsi loss (CrossEntropy vs BCEWithLogits) menjadi kunci keberhasilan dalam membedakan tugas ini dengan tugas klasifikasi emosi sebelumnya. Model mampu membedakan topik berita dengan akurasi yang baik.