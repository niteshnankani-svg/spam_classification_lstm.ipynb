## 🔗 Live Demo (Coming Soon)
A simple interface will be added using Streamlit to test real-time spam predictions.



# 📩 SMS Spam Detection using Deep Learning (CNN vs LSTM vs BERT)

This project explores multiple deep learning approaches for SMS spam detection and compares their performance on the same dataset.

---

## 🚀 Problem Statement

The goal is to classify SMS messages into:

- **0 → Ham (Not Spam)**
- **1 → Spam**

This problem is widely used in:
- email filtering systems
- fraud detection
- messaging safety systems
- customer communication filtering

---

## 📊 Dataset

The dataset contains labeled SMS messages.

- Ham: 4825
- Spam: 747

This is a slightly imbalanced dataset.

---

## ⚙️ Project Workflow

1. Data loading and exploration  
2. Label encoding (ham → 0, spam → 1)  
3. Train-test split  
4. Text preprocessing (tokenization + padding)  
5. Model training (CNN, LSTM, BERT)  
6. Model evaluation  
7. Model saving and loading  
8. Prediction on custom messages  

---

## 🧠 Models Implemented

### 🔹 1. Baseline Neural Network
- Embedding
- GlobalAveragePooling1D
- Dense layers

👉 Purpose: establish baseline performance

---

### 🔹 2. CNN (Convolutional Neural Network)
- Embedding
- Conv1D
- MaxPooling
- Dense layers

👉 Captures **local patterns / phrases** like:
- "free money"
- "click here"

---

### 🔹 3. LSTM (Long Short-Term Memory) ✅ FINAL MODEL
- Embedding
- LSTM
- Dense layers

👉 Captures **sequence and context**

---

### 🔹 4. BERT (Transformer - Experimental)
- Pretrained transformer model

👉 Captures **deep contextual meaning**, but heavy for this task

---

## 📈 Results

| Model | Accuracy | Spam Recall | Insight |
|------|----------|------------|--------|
| Baseline NN | ~95% | 0.66 | Weak spam detection |
| CNN | ~98.7% | 0.92 | Good phrase detection |
| LSTM | ~99.3% | 0.96 | Best overall performance |
| BERT | Experimental | High | Overkill for this dataset |

---

## 🏆 Final Model Selection

### ✅ LSTM was selected as the final model

### Why?
- Highest accuracy
- Strongest spam recall
- Captures sequence effectively
- Lightweight compared to BERT

---

### 🔥 Key Insight

> Although BERT is more powerful, it did not significantly outperform LSTM for this short-text classification task.  
> LSTM provided the best balance of performance and efficiency.

---

## 💾 Saved Artifacts

- `spam_lstm_model.keras` → trained model  
- `spam_tokenizer.pkl` → tokenizer  
- `spam_config.pkl` → max sequence length  

---

## 🔍 Example Prediction

```python
Input: "Congratulations! You have won a free prize!"

Output:
Spam probability: 0.9989
Prediction: Spam
