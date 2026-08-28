import asyncio
import json
import os

import click
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from tqdm import tqdm

from ordinalshap.model import InstanceModel
from ordinalshap.splitters import WordSplitter
from ordinalshap.maskers import DeleteMasker, ReplacementMasker

RESULTS_DIR = "results"
CACHE_DIR = "e3_response_cache"
DEFAULT_RESULTS_CSV = "results/results.csv"

MODELS = ["deepseek", "gemma", "mistral"]
DATASETS = ["sst-5", "yelp-20", "yelp-50"]

PAYOFF = "ordinal"
MASKERS = ["delete", "replacement"]
REPLACEMENT_TOKEN = "[REMOVED]"

MODEL_NAMES = {
  "deepseek": "deepseek-ai/DeepSeek-V4-Flash-0731",
  "gemma": "google/gemma-4-26B-A4B-it",
  "mistral": "mistralai/Mistral-Small-24B-Instruct-2501",
}

API_KEY = os.environ["DEEPINFRA_API_KEY"]
BASE_URL = "https://api.deepinfra.com/v1/openai"

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

SST5_MAPPING = {
  "very negative": 0,
  "negative": 1,
  "neutral": 2,
  "positive": 3,
  "very positive": 4,
}

YELP_MAPPING = {
  "0": 0,
  "1": 1,
  "2": 2,
  "3": 3,
  "4": 4,
}

DATASET_CONFIGS = {
  "sst5": {"system_prompt": SST5_SYSTEM_PROMPT, "mapping": SST5_MAPPING},
  "yelp": {"system_prompt": YELP_SYSTEM_PROMPT, "mapping": YELP_MAPPING},
}


def dataset_config(dataset_name):
  return DATASET_CONFIGS[dataset_name.split("-")[0]]


def ordinal_value(response, mapping):
  norm = response.strip()
  if norm in mapping:
    return float(mapping[norm])
  values = list(mapping.values())
  return (max(values) + min(values)) / 2.0


def cache_path(dataset, model, sample_id):
  os.makedirs(CACHE_DIR, exist_ok=True)
  return os.path.join(CACHE_DIR, f"{dataset}_{model}_{sample_id}.json")


def load_text_cache(dataset, model, sample_id, original_text):
  path = cache_path(dataset, model, sample_id)
  if not os.path.exists(path):
    return {}, path
  with open(path) as f:
    data = json.load(f)
  if data.get("_text") != original_text:
    return {}, path
  return data["responses"], path


def save_text_cache(path, original_text, text_cache):
  with open(path, "w") as f:
    json.dump({"_text": original_text, "responses": text_cache}, f)


def build_single_token_text(tokens, token_idx, masker):
  if masker == "delete":
    remaining = [t for i, t in enumerate(tokens) if i != token_idx]
    return " ".join(remaining)
  else:
    out = [REPLACEMENT_TOKEN if i == token_idx else t for i, t in enumerate(tokens)]
    return " ".join(out)


async def compute_deltas(wide, dataset_name, model_name):
  config = dataset_config(dataset_name)
  system_prompt = config["system_prompt"]
  mapping = config["mapping"]
  model_full_name = MODEL_NAMES[model_name]

  data = pd.read_csv(f"data/{dataset_name}.csv")
  splitter = WordSplitter()

  model = InstanceModel(
    model_name=model_full_name,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.0,
    max_tokens=5,
    max_concurrency=8,
    requests_per_minute=50,
  )

  delta_columns = {"delete_delta": {}, "replacement_delta": {}}

  sample_ids = sorted(wide["sample_id"].unique())
  n_skipped = 0
  n_attempted = 0
  for sample_id in tqdm(sample_ids):
    text = data.loc[sample_id, "text"]
    tokens = splitter.split(text)
    n_tokens = len(tokens)

    text_cache, cpath = load_text_cache(dataset_name, model_name, sample_id, text)

    if text not in text_cache:
      text_cache[text] = await model.generate(user_prompt=text, system_prompt=system_prompt)
    full_response = text_cache[text]
    full_ordinal = ordinal_value(full_response, mapping)

    features = sorted(wide[wide["sample_id"] == sample_id]["feature"].unique())
    if len(features) != n_tokens:
      n_skipped += 1
      continue
    n_attempted += 1

    texts_by_masker = {"delete": {}, "replacement": {}}
    for masker in MASKERS:
      for feature in features:
        texts_by_masker[masker][feature] = build_single_token_text(tokens, feature, masker)

    all_texts = set()
    for masker in MASKERS:
      all_texts.update(texts_by_masker[masker].values())
    to_query = [t for t in all_texts if t not in text_cache]

    if to_query:
      responses = await model.batch_generate(prompts=to_query, system_prompt=system_prompt)
      for t, r in zip(to_query, responses):
        text_cache[t] = r

    save_text_cache(cpath, text, text_cache)

    for masker, column in [("delete", "delete_delta"), ("replacement", "replacement_delta")]:
      for feature in features:
        response = text_cache[texts_by_masker[masker][feature]]
        delta = full_ordinal - ordinal_value(response, mapping)
        delta_columns[column][(sample_id, feature)] = delta

  wide = wide.copy()
  wide["delete_delta"] = wide.apply(lambda row: delta_columns["delete_delta"].get((row["sample_id"], row["feature"])), axis=1)
  wide["replacement_delta"] = wide.apply(lambda row: delta_columns["replacement_delta"].get((row["sample_id"], row["feature"])), axis=1)
  return wide, n_attempted, n_skipped


def spearman(x, y):
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  if len(x) < 2:
    return np.nan
  if np.all(x == x[0]) or np.all(y == y[0]):
    return np.nan
  return spearmanr(x, y).statistic


def sign_agreement_symmetric(x, y):
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  sx = np.sign(x)
  sy = np.sign(y)
  mask = (sx != 0) & (sy != 0)
  if mask.sum() == 0:
    return np.nan
  return np.mean(sx[mask] == sy[mask])


def sign_agreement_asymmetric(prediction, ground_truth):
  prediction = np.asarray(prediction, dtype=float)
  ground_truth = np.asarray(ground_truth, dtype=float)
  mask = ground_truth != 0
  if mask.sum() == 0:
    return np.nan
  return np.mean(np.sign(ground_truth[mask]) == np.sign(prediction[mask]))


def masker_agreement(wide):
  rows = []
  for sample_id, group in wide.groupby("sample_id"):
    delete_vals = group["delete"].values
    replacement_vals = group["replacement"].values
    rows.append({
      "sample_id": sample_id,
      "spearman": spearman(delete_vals, replacement_vals),
      "sign_agreement": sign_agreement_symmetric(delete_vals, replacement_vals),
      "n_tokens": len(group),
    })
  return pd.DataFrame(rows)


def summarize_agreement(per_sample, n_attempted):
  rows = []
  for metric in ["spearman", "sign_agreement"]:
    values = per_sample[metric].dropna()
    rows.append({
      "metric": metric,
      "mean": values.mean(),
      "std": values.std(),
      "median": values.median(),
      "n": len(values),
      "n_attempted": n_attempted,
      "coverage": (len(values) / n_attempted) if n_attempted else np.nan,
    })
  return pd.DataFrame(rows)


def fidelity_metrics(shap_values, deltas):
  shap_values = np.asarray(shap_values, dtype=float)
  deltas = np.asarray(deltas, dtype=float)
  magnitude = spearman(shap_values, deltas)
  magnitude_unsigned = spearman(np.abs(shap_values), np.abs(deltas))
  direction = sign_agreement_asymmetric(prediction=shap_values, ground_truth=deltas)
  return magnitude, magnitude_unsigned, direction


def analyze_fidelity(wide):
  rows = []
  for sample_id, group in wide.groupby("sample_id"):
    delete_shap = group["delete"].values
    delete_delta = group["delete_delta"].values
    delete_spearman, delete_spearman_unsigned, delete_sign = fidelity_metrics(delete_shap, delete_delta)
    rows.append({
      "sample_id": sample_id,
      "masker": "delete",
      "spearman_magnitude": delete_spearman,
      "spearman_magnitude_unsigned": delete_spearman_unsigned,
      "sign_agreement": delete_sign,
      "n_tokens": len(group),
    })

    replacement_shap = group["replacement"].values
    replacement_delta = group["replacement_delta"].values
    replacement_spearman, replacement_spearman_unsigned, replacement_sign = fidelity_metrics(replacement_shap, replacement_delta)
    rows.append({
      "sample_id": sample_id,
      "masker": "replacement",
      "spearman_magnitude": replacement_spearman,
      "spearman_magnitude_unsigned": replacement_spearman_unsigned,
      "sign_agreement": replacement_sign,
      "n_tokens": len(group),
    })
  return pd.DataFrame(rows)


def summarize_fidelity(per_sample, n_attempted):
  rows = []
  metrics = ["spearman_magnitude", "sign_agreement"]
  for masker in MASKERS:
    subset = per_sample[per_sample["masker"] == masker]
    for metric in metrics:
      values = subset[metric].dropna()
      rows.append({
        "masker": masker,
        "metric": metric,
        "mean": values.mean(),
        "std": values.std(),
        "median": values.median(),
        "n": len(values),
        "n_attempted": n_attempted,
        "coverage": (len(values) / n_attempted) if n_attempted else np.nan,
      })
  return pd.DataFrame(rows)


def rank_biserial_effect_size(diff):
  diff = np.asarray(diff, dtype=float)
  diff = diff[np.isfinite(diff)]
  diff = diff[diff != 0]
  if len(diff) == 0:
    return np.nan
  ranks = pd.Series(np.abs(diff)).rank(method="average").values
  w_pos = ranks[diff > 0].sum()
  w_neg = ranks[diff < 0].sum()
  denominator = w_pos + w_neg
  if denominator == 0:
    return np.nan
  return (w_pos - w_neg) / denominator


def paired_wilcoxon(per_sample, metric):
  delete = per_sample[per_sample["masker"] == "delete"].set_index("sample_id")[metric]
  replacement = per_sample[per_sample["masker"] == "replacement"].set_index("sample_id")[metric]
  common = delete.index.intersection(replacement.index)
  delete = delete.loc[common]
  replacement = replacement.loc[common]
  valid = delete.notna() & replacement.notna()
  delete = delete[valid]
  replacement = replacement[valid]
  if len(delete) < 2:
    return {
      "metric": metric,
      "masker_a": "delete",
      "masker_b": "replacement",
      "delete_mean": np.nan,
      "replacement_mean": np.nan,
      "mean_difference": np.nan,
      "median_difference": np.nan,
      "statistic": np.nan,
      "p_value": np.nan,
      "effect_size": np.nan,
      "n": len(delete),
    }

  diff = delete.values - replacement.values
  try:
    statistic, p_value = wilcoxon(delete.values, replacement.values, alternative="two-sided")
  except ValueError:
    statistic = np.nan
    p_value = np.nan

  effect = rank_biserial_effect_size(diff)
  return {
    "metric": metric,
    "masker_a": "delete",
    "masker_b": "replacement",
    "delete_mean": delete.mean(),
    "replacement_mean": replacement.mean(),
    "mean_difference": diff.mean(),
    "median_difference": np.median(diff),
    "statistic": statistic,
    "p_value": p_value,
    "effect_size": effect,
    "n": len(delete),
  }


def run_statistical_tests(per_sample, model_name, dataset_name):
  metrics = ["spearman_magnitude", "sign_agreement"]
  rows = [paired_wilcoxon(per_sample, metric) for metric in metrics]
  result = pd.DataFrame(rows)
  result["model"] = model_name
  result["dataset"] = dataset_name
  return result


def load_results(results_csv, dataset, model):
  if not os.path.exists(results_csv):
    raise FileNotFoundError(f"Could not find results file:\n{results_csv}")

  df = pd.read_csv(results_csv)
  subset = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["payoff"] == PAYOFF)].copy()

  if subset.empty:
    raise ValueError(f"No rows found for dataset={dataset!r}, model={model!r}, payoff={PAYOFF!r}.")

  maskers_found = sorted(subset["masker"].dropna().unique())
  expected = sorted(MASKERS)
  if maskers_found != expected:
    raise ValueError(f"Expected maskers {expected}, found {maskers_found}")

  required_columns = ["sample_id", "feature", "shapley"]
  for column in required_columns:
    if column not in subset.columns:
      raise ValueError(f"Missing required column: {column}")

  delete_df = subset[subset["masker"] == "delete"][["sample_id", "feature", "shapley"]].rename(columns={"shapley": "delete"})
  replacement_df = subset[subset["masker"] == "replacement"][["sample_id", "feature", "shapley"]].rename(columns={"shapley": "replacement"})
  wide = delete_df.merge(replacement_df, on=["sample_id", "feature"], how="inner")
  return wide


def print_summary(agreement_summary, fidelity_summary, statistical_tests):
  print()
  print("=" * 80)
  print("E4 MASKER AGREEMENT (mean/std + coverage)")
  print("=" * 80)
  print(agreement_summary.to_string(index=False))

  print()
  print("=" * 80)
  print("E4 FIDELITY (mean/std + coverage)")
  print("=" * 80)
  print(fidelity_summary.to_string(index=False))

  print()
  print("=" * 80)
  print("E4 WILCOXON: DELETE VS REPLACEMENT")
  print("=" * 80)
  print(statistical_tests.to_string(index=False))


async def run(model, dataset, results_csv, output_dir):
  wide = load_results(results_csv, dataset, model)
  wide, n_attempted, n_skipped = await compute_deltas(wide, dataset, model)
  print(f"Samples skipped (feature/token count mismatch): {n_skipped} / {n_attempted + n_skipped}")

  agreement = masker_agreement(wide)
  agreement.to_csv(os.path.join(output_dir, "e4_masker_agreement_per_sample.csv"), index=False)
  agreement_summary = summarize_agreement(agreement, n_attempted)
  agreement_summary.to_csv(os.path.join(output_dir, "e4_masker_agreement_summary.csv"), index=False)

  fidelity = analyze_fidelity(wide)
  fidelity.to_csv(os.path.join(output_dir, "e4_fidelity_per_sample.csv"), index=False)
  fidelity_summary = summarize_fidelity(fidelity, n_attempted)
  fidelity_summary.to_csv(os.path.join(output_dir, "e4_fidelity_summary.csv"), index=False)

  statistical_tests = run_statistical_tests(fidelity, model, dataset)
  statistical_tests.to_csv(os.path.join(output_dir, "e4_wilcoxon_raw.csv"), index=False)

  print_summary(agreement_summary, fidelity_summary, statistical_tests)


@click.command()
@click.option("--model_name", type=click.Choice(MODELS), required=True)
@click.option("--dataset_name", type=click.Choice(DATASETS), required=True)
@click.option("--results_csv", type=str, default=DEFAULT_RESULTS_CSV, show_default=True)
def main(model_name, dataset_name, results_csv):
  output_dir = os.path.join(RESULTS_DIR, "e4", f"{dataset_name}_{model_name}")
  os.makedirs(output_dir, exist_ok=True)
  asyncio.run(run(model_name, dataset_name, results_csv, output_dir))


if __name__ == "__main__":
  main()