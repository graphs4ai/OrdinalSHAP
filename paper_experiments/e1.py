import hashlib
import itertools
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types
import torch

RESULTS_DIR = "results"
output_dir = os.path.join(RESULTS_DIR, "e1")
os.makedirs(output_dir, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GOOGLE_EMBEDDING_CACHE_DIR = "google_embedding_cache"

api_key = os.environ.get("GOOGLE_API_KEY")

SCALES = {
  "likert": ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
  "frequency": ["Never", "Rarely", "Sometimes", "Often", "Always"],
  "pain": ["No pain", "Mild pain", "Moderate pain", "Severe pain", "Extreme pain"],
  "quality": ["Poor", "Fair", "Good", "Very good", "Excellent"],
  "sst-5": ["very negative", "negative", "neutral", "positive", "very positive"],
  "yelp": ["0", "1", "2", "3", "4"],
}

FORMATS = {
  "bare": lambda lab: lab,
  "sentence": lambda lab: f"The selected option is {lab}.",
}

EMBEDDERS = {
  "minilm": ("sentence-transformers/all-MiniLM-L6-v2", None),
  "bge": ("BAAI/bge-base-en-v1.5", None),
  "gte": ("thenlper/gte-base", None),
  "e5": ("intfloat/e5-base-v2", "query: "),
}

def sim_embeddings(texts, model, prefix=None):
  payload = [prefix + t for t in texts] if prefix else texts
  emb = model.encode(payload, convert_to_numpy=True, normalize_embeddings=True)
  return emb @ emb.T

def sim_tfidf(texts):
  vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
  X = vec.fit_transform(texts).toarray()
  norms = np.linalg.norm(X, axis=1, keepdims=True)
  norms[norms == 0] = 1.0
  X = X / norms
  return X @ X.T

def _bm25_tokenize(text):
  return re.findall(r"\b\w+\b", text.lower())

def sim_bm25(texts):
  tokenized = [_bm25_tokenize(t) for t in texts]
  bm25 = BM25Okapi(tokenized)
  k = len(texts)
  S = np.zeros((k, k))
  for i in range(k):
    S[i, :] = bm25.get_scores(tokenized[i])
  S = (S + S.T) / 2.0
  return S

def _cache_key(text):
  return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

def _load_cached_google_embedding(text):
  path = os.path.join(GOOGLE_EMBEDDING_CACHE_DIR, f"{_cache_key(text)}.npy")
  if os.path.exists(path):
    return np.load(path)
  return None

def _save_cached_google_embedding(text, emb):
  os.makedirs(GOOGLE_EMBEDDING_CACHE_DIR, exist_ok=True)
  path = os.path.join(GOOGLE_EMBEDDING_CACHE_DIR, f"{_cache_key(text)}.npy")
  np.save(path, emb)

def fetch_google_embeddings(all_texts, client, batch_size=100):
  unique_texts = list(dict.fromkeys(all_texts))
  embeddings = {}

  for text in unique_texts:
    cached = _load_cached_google_embedding(text)
    if cached is not None:
      embeddings[text] = cached
      continue
    result = client.models.embed_content(
      model="gemini-embedding-2-preview",
      contents=f'task: sentence similarity | query: {text}'
    )
    emb = np.array(result.embeddings[0].values, dtype=float)
    embeddings[text] = emb
    _save_cached_google_embedding(text, emb)

  return embeddings

def sim_google(texts, embedding_cache):
  emb = np.stack([embedding_cache[t] for t in texts])
  norms = np.linalg.norm(emb, axis=1, keepdims=True)
  norms[norms == 0] = 1.0
  emb = emb / norms
  return emb @ emb.T

def pair_arrays(S):
  k = S.shape[0]
  d, s = [], []
  for i, j in itertools.combinations(range(k), 2):
    d.append(abs(i - j))
    s.append(S[i, j])
  return np.asarray(d, float), np.asarray(s, float)

def violation_rate(S, tol=1e-9):
  k = S.shape[0]
  violations = 0
  total = 0
  for a, b, c in itertools.permutations(range(k), 3):
    if abs(a - b) >= abs(a - c):
      continue
    total += 1
    diff = S[a, b] - S[a, c]
    if diff < -tol:
      violations += 1
  return violations / total

def evaluate(S):
  d, s = pair_arrays(S)
  rho = spearmanr(d, s)[0]
  tau = kendalltau(d, s).statistic
  violation = violation_rate(S)
  return {
    "n_pairs": len(d),
    "spearman": rho,
    "kendall": tau,
    "violation_rate": violation,
  }

def main():
  rows = []
  combos = []
  for scale_name, labels in SCALES.items():
    for fmt_name, fmt in FORMATS.items():
      texts = [fmt(lab) for lab in labels]
      combos.append((scale_name, fmt_name, texts))

  for scale_name, fmt_name, texts in combos:
    rows.append({
      "scale": scale_name,
      "format": fmt_name,
      "encoder": "tfidf",
      **evaluate(sim_tfidf(texts)),
    })

  for scale_name, fmt_name, texts in combos:
    rows.append({
      "scale": scale_name,
      "format": fmt_name,
      "encoder": "bm25",
      **evaluate(sim_bm25(texts)),
    })

  for enc_name, (model_id, prefix) in EMBEDDERS.items():
    model = SentenceTransformer(model_id, device=DEVICE)
    for scale_name, fmt_name, texts in combos:
      S = sim_embeddings(texts, model, prefix)
      rows.append({
        "scale": scale_name,
        "format": fmt_name,
        "encoder": enc_name,
        **evaluate(S),
      })
    del model

  client = genai.Client(api_key=api_key)

  all_texts = [text for _, _, texts in combos for text in texts]
  embedding_cache = fetch_google_embeddings(all_texts, client)

  for scale_name, fmt_name, texts in combos:
    S = sim_google(texts, embedding_cache)
    rows.append({
      "scale": scale_name,
      "format": fmt_name,
      "encoder": "google-embeddings-2",
      **evaluate(S),
    })

  df = pd.DataFrame(rows)
  df.to_csv(os.path.join(output_dir, "e1_results.csv"), index=False)


if __name__ == "__main__":
  main()