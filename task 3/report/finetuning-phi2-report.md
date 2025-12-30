# Laporan Task 3: Fine-Tuning Phi-2 untuk Text Summarization (XSum)

**Repository Name:** `finetuning-phi2-summarization`
**Model:** Microsoft Phi-2 (2.7B Parameters) - 4-bit Quantized (QLoRA)
**Dataset:** XSum (Extreme Summarization)
**Task:** Abstractive Text Summarization

---

## 1. Pendahuluan

Laporan ini mendokumentasikan proses *fine-tuning* pada model **Microsoft Phi-2** untuk menyelesaikan tugas **Abstractive Summarization**. Tugas ini bertujuan untuk menghasilkan ringkasan singkat (satu kalimat) dari sebuah artikel berita yang panjang.

Pendekatan yang digunakan adalah **QLoRA (Quantized Low-Rank Adaptation)**. Teknik ini memungkinkan pelatihan model bahasa besar (LLM) pada GPU dengan VRAM terbatas (seperti T4 di Google Colab) dengan cara membekukan bobot model utama dalam presisi 4-bit dan hanya melatih adapter kecil tambahan.

## 2. Metodologi

### A. Dataset

* **Sumber Data:** `EdinburghNLP/xsum` dari Hugging Face Hub.
* **Strategi Load:** Menggunakan revisi `refs/convert/parquet` untuk akses data tabel langsung.
* **Sampel:** 1.000 data latih (subset) untuk demonstrasi efisiensi *training*.
* **Struktur Data:**
* `document`: Teks berita lengkap (BBC articles).
* `summary`: Ringkasan satu kalimat (Gold Standard).



### B. Preprocessing (Instruction Tuning)

Agar Phi-2 (yang merupakan *Base Model*) dapat mengikuti perintah meringkas, format data diubah menjadi struktur instruksi (*Alpaca Style*).

**Format Prompt:**

```text
### Instruction:
Summarize the following text:
[Isi Berita / Document]

### Response:
[Isi Ringkasan / Summary]<EOS_TOKEN>

```

**Kode Formatting:**

```python
def format_instruction(sample):
    intro = "Summarize the following text:"
    return f"### Instruction:\n{intro}\n{sample['document']}\n\n### Response:\n{sample['summary']}{tokenizer.eos_token}"

```

### C. Konfigurasi Model & Training

* **Arsitektur:** `PhiForCausalLM` dengan `BitsAndBytesConfig` (4-bit NF4 Quantization).
* **LoRA Config (Critical Fix):**
* *Target Modules:* `["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]`. (Konfigurasi ini krusial agar model Phi-2 benar-benar "belajar" dan loss menurun).
* *Rank (r):* 32
* *Alpha:* 64


* **Training Arguments:**
* Epochs: 1
* Learning Rate: 2e-4
* Optimizer: `paged_adamw_32bit`
* Patch Khusus T4: Memaksa parameter LoRA ke `float32` untuk menghindari error `unscale FP16 gradients`.



## 3. Hasil Implementasi

### Proses Training

Training berjalan selama 250 steps (1 epoch). Terlihat penurunan *Loss* yang signifikan, menandakan model berhasil beradaptasi dari sekadar *text completion* menjadi *summarizer*.

* **Initial Loss:** ~2.41
* **Final Loss:** 1.89

*(Grafik loss menunjukkan tren menurun yang stabil tanpa spike yang drastis)*

### Evaluasi Interaktif (Inference)

Pengujian dilakukan dengan parameter generasi yang mencegah repetisi (*mumbling*) dan halusinasi berlebih.

**Parameter Inference:**

* `temperature`: 0.5 (Lebih deterministik/fokus).
* `repetition_penalty`: 1.2.
* `max_new_tokens`: 100.

**Contoh Input (Berita Konflik Ethiopia-Eritrea):**

> *"Ethiopia has not commented on the reported fighting in the Tsorona area... Residents on the Ethiopian side of the border reported hearing gunfire and seeing a large movement of troops..."*

## 4. Analisis Hasil

Dari pengujian data ke-10, didapatkan perbandingan sebagai berikut:

* **Ground Truth (Asli):**
*"Eritrea has accused Ethiopia of launching an attack at the countries' heavily-militarised border."*
* **Prediksi Model (Phi-2 Finetuned):**
*"An apparent clash between soldiers from neighbouring Ethiopia and Eritrea has left one person dead near their shared border."*

**Analisis:**

1. **Struktur Bahasa:** Model menghasilkan kalimat bahasa Inggris yang valid secara gramatikal dan koheren (SPOK jelas). Tidak ada lagi *output* acak atau repetisi kata ("Instructionalgorithm", dll) yang terjadi sebelum perbaikan *target modules*.
2. **Pemahaman Konteks (Abstractive):** Model berhasil menyimpulkan kata kunci "gunfire", "troops", dan "fighting" menjadi satu kata **"clash"**. Ini menunjukkan kemampuan *abstractive summarization* (membuat kalimat baru), bukan sekadar *extractive* (mengambil potongan kalimat asli).
3. **Akurasi Fakta:** Terdapat sedikit halusinasi (*"left one person dead"*), yang kemungkinan merupakan bias model terhadap berita konflik militer, namun secara kontekstual ringkasannya sangat relevan dengan topik berita.

## 5. Kesimpulan

Proses *fine-tuning* Phi-2 menggunakan metode QLoRA pada GPU T4 berhasil dilaksanakan.

1. **Isu Teknis Teratasi:** Masalah kompatibilitas tipe data pada GPU T4 dan kesalahan penamaan layer (`Wqkv` vs `q_proj`) berhasil diperbaiki.
2. **Kualitas Model:** Dengan Loss akhir **1.89**, model menunjukkan peningkatan kemampuan drastis dibandingkan kondisi awal. Model mampu menghasilkan ringkasan yang mudah dibaca dan relevan secara topik.

Langkah selanjutnya untuk improvisasi adalah memperbesar jumlah dataset (full epoch) dan melakukan *hyperparameter tuning* untuk mengurangi tingkat halusinasi fakta.