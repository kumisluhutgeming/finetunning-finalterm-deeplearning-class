
# Laporan Task 2: Fine-Tuning T5 untuk Question Answering (SQuAD)

**Repository Name:** `finetuning-t5-qa`  
**Model:** T5-Base (Text-to-Text Transfer Transformer)  
**Dataset:** Stanford Question Answering Dataset (SQuAD)  
**Task:** Generative Question Answering  

---

## 1. Pendahuluan
Laporan ini mendokumentasikan implementasi fine-tuning pada model **T5-Base** untuk menyelesaikan tugas **Question Answering (QA)**. Berbeda dengan pendekatan ekstraktif tradisional (seperti BERT yang memprediksi indeks awal dan akhir jawaban), pendekatan ini menggunakan paradigma **Generative QA** atau *Text-to-Text*.

Dalam pendekatan ini, model diberikan input berupa gabungan pertanyaan dan konteks bacaan, kemudian dilatih untuk **mengenerate (menulis ulang)** jawaban yang tepat secara tekstual. Dataset yang digunakan adalah **SQuAD**, standar emas dalam evaluasi pemahaman bacaan mesin.

## 2. Metodologi

### A. Dataset
- **Sumber Data:** `squad` dari Hugging Face Hub.
- **Struktur Data:** Terdiri dari pasangan `context` (paragraf bacaan), `question` (pertanyaan), dan `answers` (teks jawaban beserta posisi start-nya).
- **Strategi Sampling:** Menggunakan subset data (2.000 sampel) untuk mempercepat proses demonstrasi training.

### B. Preprocessing (Format Text-to-Text)
Karena T5 adalah model *Sequence-to-Sequence*, format input data harus diubah menjadi string tunggal dengan prefix khusus agar model memahami tugas yang diberikan.

- **Format Input:** `"question: [Pertanyaan] context: [Konteks Bacaan]"`
- **Format Target (Label):** `"[Teks Jawaban]"`

**Kode Preprocessing:**
```python
input_text = f"question: {question} context: {context}"
model_inputs = tokenizer(input_text, max_length=512, truncation=True, padding="max_length")

```

### C. Konfigurasi Model & Training

* **Arsitektur:** `AutoModelForSeq2SeqLM` menggunakan checkpoint `t5-base`.
* **Optimizer:** `Seq2SeqTrainer` dengan dukungan `fp16=True` (Mixed Precision) untuk efisiensi memori GPU.
* **Hyperparameter:**
* Epochs: 3
* Batch Size: 8
* Learning Rate: 2e-5
* Generation Config: Menggunakan `predict_with_generate=True` saat evaluasi.



## 3. Hasil Implementasi

### Proses Training

Model dilatih selama 3 epoch. Penurunan *Training Loss* menunjukkan bahwa model berhasil mempelajari hubungan antara pertanyaan+konteks dengan jawaban targetnya.

*(Silakan tempel screenshot grafik Loss dari notebook di sini)*

### Evaluasi Interaktif (Inference)

Pengujian dilakukan menggunakan metode **Beam Search** (`num_beams=4`) untuk menghasilkan teks jawaban yang paling mungkin dan koheren.

**Skenario Pengujian:**

* **Context:** Artikel singkat mengenai sejarah *game* "Super Mario Bros" (dikembangkan oleh Nintendo, dirilis 1985).
* **Question:** *"Who developed Super Mario Bros?"*

**Kode Inference:**

```python
outputs = model.generate(
    inputs["input_ids"], 
    max_length=32, 
    num_beams=4,    
    early_stopping=True
)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

```

## 4. Analisis Hasil

Dari pengujian manual, didapatkan hasil sebagai berikut:

* **Pertanyaan:** "Who developed Super Mario Bros?"
* **Prediksi Model:** "Nintendo"
* **Analisis:** Model berhasil memahami konteks kalimat *"Super Mario Bros. is a platform game developed and published by Nintendo"*. Alih-alih hanya menunjuk posisi kata, T5 berhasil mengenerate kata "Nintendo" sebagai output teks yang valid. Ini membuktikan bahwa model memahami semantik "Who developed" dan mengaitkannya dengan entitas "Nintendo" di dalam teks.

## 5. Kesimpulan

Implementasi *Generative Question Answering* menggunakan T5-Base berhasil dilakukan. Pendekatan *Text-to-Text* terbukti efektif dan lebih fleksibel dibandingkan pendekatan ekstraktif murni. Model mampu "membaca" materi yang diberikan (Context) dan "menjawab" ujian (Question) dengan akurat sesuai materi tersebut.
