# Laporan Task 1: Fine-Tuning BERT untuk Klasifikasi Teks (GoEmotions)

**Model:** BERT Base Uncased  
**Dataset:** Google Research - GoEmotions  
**Task:** Multi-Label Text Classification  

---

## 1. Pendahuluan
Proyek ini bertujuan untuk melakukan fine-tuning pada model pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** agar dapat mendeteksi emosi dalam teks bahasa Inggris. Dataset yang digunakan adalah **GoEmotions**, yang memiliki karakteristik unik berupa *Multi-Label Classification*, di mana satu kalimat dapat memiliki lebih dari satu label emosi (total 28 label, seperti *Admiration, Joy, Sadness, dll*).

## 2. Metodologi

### A. Dataset dan Preprocessing
- **Sumber Data:** Dataset diambil dari Hugging Face Hub (`google-research-datasets/go_emotions`, konfigurasi `simplified`).
- **Tokenisasi:** Menggunakan `AutoTokenizer` dari `bert-base-uncased` dengan `max_length=128`.
- **Handling Multi-Label:** Karena ini adalah klasifikasi multi-label, format label diubah dari list index menjadi vektor **One-Hot Encoding** (Multi-Hot) dengan tipe data `float32`.
  
  *Contoh transformasi label:*
  - Input: `[3, 27]` (Labels: Anger, Neutral)
  - Output: `[0.0, 0.0, 0.0, 1.0, ..., 1.0]` (Vector sepanjang 28 dimensi)

### B. Konfigurasi Model
- **Arsitektur:** `BertForSequenceClassification`
- **Jumlah Label:** 28
- **Loss Function:** Menggunakan `BCEWithLogitsLoss` (Binary Cross Entropy) yang diaktifkan otomatis dengan parameter `problem_type="multi_label_classification"`.
- **Hyperparameter:**
  - Epochs: 3
  - Batch Size: 8
  - Learning Rate: 2e-5

## 3. Implementasi dan Training
Proses training dilakukan menggunakan `Trainer` API dari Hugging Face.

### Kode Setup Model:
```python
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=28,
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id
)
```

**Grafik Training Loss:**
Selama 3 epoch, model menunjukkan penurunan loss yang signifikan, menandakan model berhasil mempelajari pola emosi dari data training tanpa mengalami overfitting yang parah.

![visualisasi-training-mnli](visualisasi-training-mnli.png)

## 4. Hasil dan Evaluasi
Evaluasi dilakukan menggunakan metrik F1-Score (Micro Average) karena adanya ketidakseimbangan kelas pada dataset emosi.

- Threshold Prediksi: 0.5 (Jika probabilitas > 50%, label dianggap aktif).
- Hasil Akhir: Model mampu memprediksi kombinasi emosi yang kompleks.
- Pengujian Inference (Contoh Kasus):
> - Input:
> > "I am so happy that I finally got the promotion, but a bit nervous about the new responsibility."

> - Prediksi Model:

> > Joy: (Probabilitas tinggi) -> Merespon kata "happy" dan "promotion".

> > Nervousness: (Probabilitas tinggi) -> Merespon kata "nervous".

## 5. Kesimpulan
Model BERT berhasil di-finetune untuk menangani tugas Multi-Label Classification. Penyesuaian tipe data label ke Float32 dan penggunaan Sigmoid activation sangat krusial dalam keberhasilan training ini.