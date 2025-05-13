
# 🧠 Financial News Semantic Search & Sentiment Analysis

This project enables semantic search and sentiment analysis on Indian financial news data. It integrates text summarization, embedding-based retrieval, classification, and explainable AI (LIME).

---

## 🚀 Features

- **Semantic Search**: FAISS-based retrieval using sentence-transformers
- **Summarization**: T5-small model generates article summaries
- **Sentiment Analysis**: Custom fine-tuned BERT model
- **Explainability**: LIME highlights the key contributors to sentiment
- **NLP Pipeline**: Includes NLTK preprocessing and spaCy lemmatization

---

## 📂 Dataset

The input dataset should be a CSV file with at least the following columns:
- `Title`
- `Description`

Example: `IndianFinancialNews.csv`

---

## 🛠️ Setup

Install required packages:

```bash
pip install pandas numpy faiss-cpu sentence-transformers transformers spacy scikit-learn torch lime nltk
python -m spacy download en_core_web_sm
```

Download NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🧪 How It Works

1. **Text Embeddings**:
    - Embeds news descriptions using `all-MiniLM-L6-v2`.

2. **Semantic Search**:
    - Uses FAISS index to find top-k similar documents.

3. **Summarization**:
    - Applies T5-small transformer for summaries.

4. **Sentiment Analysis**:
    - Uses a fine-tuned classifier to predict sentiment.

5. **LIME Explanation**:
    - Explains sentiment predictions with local interpretable highlights.

---

## 📦 Output

For each matched result:
- Title
- Summary
- Sentiment & Confidence
- Key word contributions via LIME

---

## 📜 License

MIT License

---

## 🤖 Models Used

- [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [`t5-small`](https://huggingface.co/t5-small)
- [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny)
