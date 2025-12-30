# 🧠 NLP Fine-Tuning Experiments

This repository contains a collection of Natural Language Processing (NLP) fine-tuning projects. The goal is to explore various state-of-the-art architectures (BERT, T5, Phi-2) across different downstream tasks, ranging from classification to generative summarization.

## 📂 Project Overview

The repository is organized into the following tasks:

### 1. Text Classification & Sentiment Analysis

* **Models:** BERT (Bidirectional Encoder Representations from Transformers)
* **Datasets:**
* `AG News`: Topic classification (World, Sports, Business, Sci/Tech).
* `GoEmotions`: Fine-grained emotion classification (Multi-label).


* **Goal:** To adapt pre-trained BERT models for identifying topics in news articles and detecting subtle emotions in text.

### 2. Natural Language Inference (NLI)

* **Model:** BERT
* **Dataset:** MNLI (Multi-Genre Natural Language Inference)
* **Goal:** To train the model to determine the relationship between two sentences: *Entailment*, *Contradiction*, or *Neutral*.

### 3. Question Answering (QA)

* **Model:** T5 (Text-to-Text Transfer Transformer)
* **Dataset:** SQuAD (Stanford Question Answering Dataset)
* **Goal:** To fine-tune a sequence-to-sequence model that can extract or generate accurate answers based on a given context paragraph.

### 4. Abstractive Text Summarization

* **Model:** Microsoft Phi-2 (2.7B)
* **Method:** QLoRA (Quantized Low-Rank Adaptation) - 4-bit Quantization
* **Dataset:** XSum (Extreme Summarization)
* **Goal:** To perform resource-efficient fine-tuning on a Large Language Model (LLM) to generate concise, one-sentence summaries of news articles.

---

## 💾 Model Checkpoints

Due to the large size of the fine-tuned models, the checkpoints are hosted externally. You can download the weights from the links below:

```bash
https://drive.google.com/drive/folders/1gIGG7RjuGtkdAXXz33AitD7U3YhzBCzY?usp=sharing
```

| Task | Model Architecture | Dataset | 
| --- | --- | --- |
| **Emotion Classification** | BERT Base | GoEmotions | 
| **News Classification** | BERT Base | AG News | 
| **NLI** | BERT Base | MNLI | 
| **Question Answering** | T5 Base/Small | SQuAD | 
| **Summarization** | Phi-2 (QLoRA) | XSum | 

> **Note:** For the Phi-2 model, ensure you have `peft` and `bitsandbytes` installed to load the QLoRA adapters.

---

## 🛠️ Installation & Requirements

To replicate the experiments or run inference, install the required dependencies:

```bash
pip install torch transformers datasets scikit-learn pandas

```

For the **Phi-2 Summarization** task (QLoRA), additional libraries are required:

```bash
pip install peft bitsandbytes accelerate trl

```

## 🚀 Usage

Each task has its own Jupyter Notebook (`.ipynb`) containing the training pipeline, evaluation, and inference scripts.

**Example Inference (Summarization with Phi-2):**

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Base Model & Adapter
base_model = "microsoft/phi-2"
adapter_path = "path/to/downloaded/phi2-adapter"

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Generate
prompt = "### Instruction:\nSummarize this...\n..."
# ... (See notebook for full inference code)

```

## 📊 Results Highlights

* **BERT (AG News):** Achieved high accuracy in distinguishing news categories.
* **T5 (SQuAD):** Successfully learned to locate and extract answers from contexts.
* **Phi-2 (XSum):** Demonstrated the ability to perform abstractive summarization with coherent grammar and context understanding using only 4-bit precision training.

---

# **Author:**
- Bayu Setyo P
- M Afandi
- M Harits

---

Semoga projectnya lancar jaya bang! 🔥
