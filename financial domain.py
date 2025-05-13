import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

saved_model_dir = "saved_model"
sentiment_model = AutoModelForSequenceClassification.from_pretrained(saved_model_dir)
sentiment_tokenizer = AutoTokenizer.from_pretrained(saved_model_dir)
summary_model = pipeline("summarization", model="t5-small")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")
data = pd.read_csv("IndianFinancialNews.csv")
data["Description"] = data["Description"].fillna("No Description Available")

model_name = "prajjwal1/bert-tiny"
# model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def clean_text(text):
    return text.replace("\n", " ").replace("\r", " ").strip()

data["Description"] = data["Description"].apply(clean_text)
data_embeddings = embedder.encode(data["Description"].tolist(), convert_to_tensor=False)
data_embeddings = np.array(data_embeddings, dtype="float32")

index = faiss.IndexFlatL2(data_embeddings.shape[1])
index.add(data_embeddings)



def summarize_text(text, max_length=50):
    summary = summary_model(text, max_length=max_length, min_length=10, do_sample=False)
    return summary[0]['summary_text']

def analyze_sentiment(summary):
    inputs = sentiment_tokenizer(summary, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = sentiment_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    sentiment = "positive" if torch.argmax(probabilities) == 1 else "negative"
    confidence = probabilities[0, torch.argmax(probabilities)].item()

    return sentiment, confidence

explainer = LimeTextExplainer(class_names=["negative", "positive"])

def predict_proba(texts):
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, padding=True, max_length=64
    )
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    return probabilities

def analyze_sentiment_with_lime(summary):
    lime_results = explain_sentiment_lime(summary)
    
    sentiment = "positive" if lime_results[0][1] > 0 else "negative"
    print(f"Sentiment: {sentiment}")
    print("Top contributing words/phrases to sentiment:")
    
    for word, contribution in lime_results:
        print(f"{word}: {contribution:.4f}")

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def explain_sentiment_lime(text):
    processed_text = preprocess_text(text)
    
    explanation = explainer.explain_instance(processed_text, predict_proba, num_features=3)
    lime_results = explanation.as_list()
    
    return lime_results

def search_financial_term(term, top_k=5):
    # Embed the query
    query_embedding = embedder.encode([term], convert_to_tensor=False)
    query_embedding = np.array(query_embedding, dtype="float32")
    
    # Search FAISS for top-k results
    distances, indices = index.search(query_embedding, k=top_k)
    
    # Retrieve documents
    results = []
    for idx in indices[0]:
        row = data.iloc[idx]
        summary = summarize_text(row["Description"], max_length=100)
        sentiment, confidence = analyze_sentiment(summary)
        contributions = explain_sentiment_lime(summary)
        
        results.append({
            "Title": row.get("Title", "N/A"),
            "Summary": summary,
            "Sentiment": sentiment,
            "Confidence": confidence,
            "Contributions": contributions
        })

    return results

if __name__ == "__main__":
    query = "reliance"
    results = search_financial_term(query)

    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Title: {result['Title']}")
        print(f"Summary: {result['Summary']}")
        print(f"Sentiment: {result['Sentiment']} (Confidence: {result['Confidence']:.2f})")
        print("Key Contributions to Sentiment:")
        for word, score in result['Contributions']:
            print(f"  {word}: {score:.2f}")