import hashlib
import os
import re
from abc import ABC, abstractmethod

import numpy as np


class Payoff(ABC):

  def fit(self, responses):
    return self

  @abstractmethod
  def evaluate(self, baseline_response, coalition_response):
    ...


class OrdinalPayoff(Payoff):

  def __init__(self, mapping, fallback_value=None):
    self.mapping = mapping
    self.fallback_value = fallback_value

  def evaluate(self, baseline_response, coalition_response):
    response = coalition_response.strip()

    if response not in self.mapping:
      if self.fallback_value is not None:
        return float(self.fallback_value)
      values = list(self.mapping.values())
      midpoint = (max(values) + min(values)) / 2.0
      return float(midpoint)

    return float(self.mapping[response])


class TFIDFPayoff(Payoff):

  def __init__(self):
    self.vectorizer = None
    self._vectors = {}

  def fit(self, responses):
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = list(responses)
    self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    matrix = self.vectorizer.fit_transform(texts).toarray()

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    self._vectors = {}
    for text, vec in zip(texts, matrix):
      self._vectors.setdefault(text, vec)
    return self

  def evaluate(self, baseline_response, coalition_response):
    if self.vectorizer is None:
      raise RuntimeError("TFIDFPayoff.fit() must be called before evaluate().")

    v1 = self._vectors.get(baseline_response)
    v2 = self._vectors.get(coalition_response)

    if v1 is None or v2 is None:
      raise KeyError("Response not seen during fit().")
    return float(v1 @ v2)


def _bm25_tokenize(text):
  return re.findall(r"\b\w+\b", text.lower())


class BM25Payoff(Payoff):

  def __init__(self, tokenizer=None):
    self._tokenizer = tokenizer or _bm25_tokenize
    self.bm25 = None
    self._texts = []
    self._doc_tokens = []

  def fit(self, responses):
    from rank_bm25 import BM25Okapi

    self._texts = list(responses)
    self._doc_tokens = [self._tokenizer(t) for t in self._texts]
    self.bm25 = BM25Okapi(self._doc_tokens)
    return self

  def evaluate(self, baseline_response, coalition_response):
    if self.bm25 is None:
      raise RuntimeError("BM25Payoff.fit() must be called before evaluate().")
    if baseline_response not in self._texts or coalition_response not in self._texts:
      raise KeyError("Response not seen during fit().")

    forward = self.bm25.get_scores(self._tokenizer(baseline_response))
    backward = self.bm25.get_scores(self._tokenizer(coalition_response))

    i = self._texts.index(coalition_response)
    j = self._texts.index(baseline_response)
    return float((forward[i] + backward[j]) / 2.0)


class SentenceEmbedder:

  def __init__(self, model_name, device=None, prefix=None):
    from sentence_transformers import SentenceTransformer

    self.model = SentenceTransformer(model_name, device=device)
    self.prefix = prefix

  def encode(self, texts):
    payload = [self.prefix + t for t in texts] if self.prefix else texts
    return self.model.encode(payload, normalize_embeddings=True, convert_to_numpy=True)


class EmbeddingPayoff(Payoff):

  def __init__(self, embedder):
    self.embedder = embedder
    self.cache = {}

  def _embedding(self, text):
    if text not in self.cache:
      self.cache[text] = self.embedder.encode([text])[0]
    return self.cache[text]

  def evaluate(self, baseline_response, coalition_response):
    e1 = self._embedding(baseline_response)
    e2 = self._embedding(coalition_response)
    return float(e1 @ e2)


def _google_cache_key(text):
  return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


class DiskEmbeddingCache:

  def __init__(self, cache_dir):
    self.cache_dir = cache_dir

  def _path(self, text):
    return os.path.join(self.cache_dir, f"{_google_cache_key(text)}.npy")

  def get(self, text):
    path = self._path(text)
    if os.path.exists(path):
      return np.load(path)
    return None

  def set(self, text, vector):
    os.makedirs(self.cache_dir, exist_ok=True)
    np.save(self._path(text), np.asarray(vector, dtype=np.float64))


class GoogleEmbedder:

  MODEL_ID = "gemini-embedding-2-preview"
  TASK_PREFIX = "task: sentence similarity | query: "

  def __init__(self, client):
    self.client = client

  def embed(self, text):
    result = self.client.models.embed_content(
      model=self.MODEL_ID,
      contents=self.TASK_PREFIX + text,
    )
    return np.array(result.embeddings[0].values, dtype=float)


class GoogleEmbeddingPayoff(Payoff):

  def __init__(self, google_embedder, cache_dir):
    self.embedder = google_embedder
    self.cache = DiskEmbeddingCache(cache_dir)
    self._mem_cache = {}

  def _embedding(self, text):
    if text in self._mem_cache:
      return self._mem_cache[text]

    vec = self.cache.get(text)
    if vec is None:
      vec = self.embedder.embed(text)
      self.cache.set(text, vec)

    norm = np.linalg.norm(vec)
    normalized = vec / norm if norm > 0 else vec
    self._mem_cache[text] = normalized
    return normalized

  def evaluate(self, baseline_response, coalition_response):
    e1 = self._embedding(baseline_response)
    e2 = self._embedding(coalition_response)
    return float(e1 @ e2)