import asyncio
import csv
import fcntl
import os
from google import genai
import click
from tqdm import tqdm
import pandas as pd

from ordinalshap.payoffs import (
  OrdinalPayoff, 
  TFIDFPayoff, 
  BM25Payoff,
  EmbeddingPayoff, 
  SentenceEmbedder,
  GoogleEmbeddingPayoff, 
  GoogleEmbedder
)
from ordinalshap.coalition_evaluator import CoalitionEvaluator
from ordinalshap.coalitions import PermutationSampler
from ordinalshap.maskers import DeleteMasker, ReplacementMasker
from ordinalshap.model import InstanceModel
from ordinalshap.shapley import ShapleyExplainer
from ordinalshap.splitters import WordSplitter

CACHE_DIR = "response_cache"
GOOGLE_EMBEDDING_CACHE_DIR = "google_embedding_cache"

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_OUT_CSV = "results.csv"

RESULT_COLUMNS = ["dataset", "model", "masker", "payoff", "sample_id", "feature", "token", "shapley"]

MODEL_NAMES = {
  "deepseek": "deepseek-ai/DeepSeek-V4-Flash-0731",
  "gemma": "google/gemma-4-26B-A4B-it",
  "mistral": "mistralai/Mistral-Small-24B-Instruct-2501",
}
MODELS = list(MODEL_NAMES)

MASKERS = ["delete", "replacement"]
PAYOFFS = ["ordinal", "tf-idf", "minilm", "bm25", "google-embeddings-2", "bge", "gte", "e5"]
DATASETS = ["sst-5", "yelp-20", "yelp-50"]

SST5_SYSTEM_PROMPT = """
You are performing a sentiment classification task. Your task is to classify the sentiment of the provided text. You MUST always select exactly one of the following categories:

very negative
negative
neutral
positive
very positive

The input text may be incomplete or contain [REMOVED] tokens. Even when the available information is limited, you MUST choose the category that is most appropriate based on the available text.

Do not mention that information is missing. Output ONLY one category: very negative, negative, neutral, positive, or very positive
"""

SST5_ORDINAL_MAPPING = {
  "very negative": 0,
  "negative": 1,
  "neutral": 2,
  "positive": 3,
  "very positive": 4,
}

YELP_SYSTEM_PROMPT = """
You are performing a sentiment classification task. Your task is to classify the sentiment of the provided text. You MUST always select exactly one of the following categories:

0 = very negative
1 = negative
2 = neutral
3 = positive
4 = very positive

The input text may be incomplete or contain [REMOVED] tokens. Even when the available information is limited, you MUST choose the category that is most appropriate based on the available text.

Do not refuse to classify. Do not explain your answer. Do not mention that information is missing. Output ONLY one integer: 0, 1, 2, 3, or 4
"""

YELP_ORDINAL_MAPPING = {
  "0": 0,
  "1": 1,
  "2": 2,
  "3": 3,
  "4": 4,
}

DATASET_CONFIGS = {
  "sst5": {"system_prompt": SST5_SYSTEM_PROMPT, "ordinal_mapping": SST5_ORDINAL_MAPPING},
  "yelp": {"system_prompt": YELP_SYSTEM_PROMPT, "ordinal_mapping": YELP_ORDINAL_MAPPING},
}


def dataset_config(dataset_name):
  prefix = dataset_name.split("-")[0]
  return DATASET_CONFIGS[prefix]


def cache_path(model_name, masker_name, n_samples, dataset_name, text_id, sampler_name="permutation"):
  os.makedirs(CACHE_DIR, exist_ok=True)
  fname = f"{dataset_name}_{model_name}_{masker_name}_{sampler_name}_{n_samples}_{text_id}.json"
  return os.path.join(CACHE_DIR, fname)


def _require_env(var_name):
  value = os.environ.get(var_name)
  if not value:
    raise RuntimeError(f"{var_name} is not set. Export it before running this script.")
  return value


def append_results(rows, out_csv):
  if not rows:
    return
  out_dir = os.path.dirname(out_csv)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)

  with open(out_csv, "a", newline="", encoding="utf-8") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    try:
      write_header = os.fstat(f.fileno()).st_size == 0
      writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
      if write_header:
        writer.writeheader()
      for row in rows:
        writer.writerow({k: row[k] for k in RESULT_COLUMNS})
      f.flush()
    finally:
      fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def build_payoff(payoff_name, ordinal_mapping):
  if payoff_name == "tf-idf":
    return TFIDFPayoff()
  if payoff_name == "minilm":
    return EmbeddingPayoff(embedder=SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2"))
  if payoff_name == "bge":
    return EmbeddingPayoff(embedder=SentenceEmbedder(model_name="BAAI/bge-base-en-v1.5"))
  if payoff_name == "gte":
    return EmbeddingPayoff(embedder=SentenceEmbedder(model_name="thenlper/gte-base"))
  if payoff_name == "e5":
    return EmbeddingPayoff(embedder=SentenceEmbedder(model_name="intfloat/e5-base-v2", prefix="query: "))
  if payoff_name == "bm25":
    return BM25Payoff()
  if payoff_name == "google-embeddings-2":
    client = genai.Client(api_key=_require_env("GOOGLE_API_KEY"))
    return GoogleEmbeddingPayoff(GoogleEmbedder(client), cache_dir=GOOGLE_EMBEDDING_CACHE_DIR)
  return OrdinalPayoff(ordinal_mapping)


async def run_dataset(model_name, masker_name, payoff_name, n_samples, dataset_name, out_csv):
  config = dataset_config(dataset_name)
  system_prompt = config["system_prompt"]
  ordinal_mapping = config["ordinal_mapping"]

  df = pd.read_csv(f"data/{dataset_name}.csv")
  model_full_name = MODEL_NAMES[model_name]

  model = InstanceModel(
    model_name=model_full_name,
    api_key=_require_env("DEEPINFRA_API_KEY"),
    base_url=DEEPINFRA_BASE_URL,
    temperature=0.0,
    max_tokens=5,
    max_concurrency=8,
    requests_per_minute=50,
  )

  splitter = WordSplitter()
  masker = DeleteMasker() if masker_name == "delete" else ReplacementMasker(replacement="[REMOVED]")
  payoff = build_payoff(payoff_name, ordinal_mapping)
  sampler = PermutationSampler(n_samples=n_samples)

  for text_id, text in tqdm(enumerate(df.text.values), total=len(df)):
    tokens = splitter.split(text)

    evaluator = CoalitionEvaluator(
      model=model,
      splitter=splitter,
      masker=masker,
      payoff=payoff,
      system_prompt=system_prompt,
      text=text,
    )

    cpath = cache_path(model_name, masker_name, n_samples, dataset_name, text_id)

    if os.path.exists(cpath):
      evaluator.load_responses(cpath, len(tokens))
      if evaluator.baseline_response is None:
        await evaluator.initialize()
    else:
      await evaluator.initialize()

    permutations = sampler.sample(len(tokens))
    explainer = ShapleyExplainer(evaluator=evaluator)
    phi = await explainer.explain(permutations=permutations)

    rows = [{
      "dataset": dataset_name,
      "model": model_name,
      "masker": masker_name,
      "payoff": payoff_name,
      "sample_id": text_id,
      "feature": feature,
      "token": tokens[feature],
      "shapley": value,
      }
      for feature, value in phi.items()
    ]

    if not os.path.exists(cpath):
      evaluator.save_responses(cpath)

    append_results(rows, out_csv)


@click.command()
@click.option("--model_name", type=click.Choice(MODELS), required=True)
@click.option("--masker_name", type=click.Choice(MASKERS), required=True)
@click.option("--payoff_name", type=click.Choice(PAYOFFS), required=True)
@click.option("--n_samples", type=int, default=10)
@click.option("--dataset_name", type=click.Choice(DATASETS), required=True)
@click.option("--out_csv", type=str, default="results/results.csv")
def main(model_name, masker_name, payoff_name, n_samples, dataset_name, out_csv):
  asyncio.run(run_dataset(model_name, masker_name, payoff_name, n_samples, dataset_name, out_csv))

if __name__ == "__main__":
  main()