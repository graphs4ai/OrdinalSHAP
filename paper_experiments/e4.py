import asyncio
import json
import os

import click
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from tqdm import tqdm

from ordinalshap.model import *
from ordinalshap.splitters import *
from ordinalshap.maskers import *

RESULTS_DIR = "results"
CACHE_DIR = "e3_response_cache"
DEFAULT_RESULTS_CSV = "results/results.csv"

MODELS = ["deepseek", "gemma", "mistral"]
DATASETS = ["sst-5", "yelp-20", "yelp-50"]

PAYOFFS = ["ordinal-directional", "ordinal-preservation"]
MASKERS = ["delete", "replacement"]
REPLACEMENT_TOKEN = "[REMOVED]"

PAYOFF_TARGET = {
  "ordinal-directional": "dir",
  "ordinal-preservation": "pres",
}

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
  "sst": {"system_prompt": SST5_SYSTEM_PROMPT, "mapping": SST5_MAPPING},
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
  span = float(max(mapping.values()) - min(mapping.values()))
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

  delta_columns = {
    "delete_delta_dir": {}, "replacement_delta_dir": {},
    "delete_delta_pres": {}, "replacement_delta_pres": {},
  }

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

    for masker in MASKERS:
      for feature in features:
        response = text_cache[texts_by_masker[masker][feature]]
        coalition_ordinal = ordinal_value(response, mapping)
        delta_columns[f"{masker}_delta_dir"][(sample_id, feature)] = full_ordinal - coalition_ordinal
        delta_columns[f"{masker}_delta_pres"][(sample_id, feature)] = abs(full_ordinal - coalition_ordinal) / span

  wide = wide.copy()
  for column, values in delta_columns.items():
    wide[column] = wide.apply(lambda row: values.get((row["sample_id"], row["feature"])), axis=1)
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


def analyze_fidelity(wide, target):
  rows = []
  for sample_id, group in wide.groupby("sample_id"):
    for masker in MASKERS:
      shap_values = group[masker].values
      deltas = group[f"{masker}_delta_{target}"].values
      magnitude, magnitude_unsigned, sign = fidelity_metrics(shap_values, deltas)
      rows.append({
        "sample_id": sample_id,
        "masker": masker,
        "target": target,
        "spearman_magnitude": magnitude,
        "spearman_magnitude_unsigned": magnitude_unsigned,
        "sign_agreement": sign,
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
    return np.nan, 0
  ranks = pd.Series(np.abs(diff)).rank(method="average").values
  w_pos = ranks[diff > 0].sum()
  w_neg = ranks[diff < 0].sum()
  denominator = w_pos + w_neg
  if denominator == 0:
    return np.nan, 0
  return (w_pos - w_neg) / denominator, len(diff)


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
      "n_nonzero": 0,
      "min_attainable_p": np.nan,
    }

  diff = delete.values - replacement.values
  try:
    statistic, p_value = wilcoxon(delete.values, replacement.values, alternative="two-sided")
  except ValueError:
    statistic = np.nan
    p_value = np.nan

  effect, n_nonzero = rank_biserial_effect_size(diff)
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
    "n_nonzero": n_nonzero,
    "min_attainable_p": 2.0 / (2 ** n_nonzero) if 0 < n_nonzero < 60 else 0.0,
  }


def run_statistical_tests(per_sample, model_name, dataset_name, payoff):
  metrics = ["spearman_magnitude", "sign_agreement"]
  rows = [paired_wilcoxon(per_sample, metric) for metric in metrics]
  result = pd.DataFrame(rows)
  result["model"] = model_name
  result["dataset"] = dataset_name
  result["payoff"] = payoff
  result["target"] = PAYOFF_TARGET[payoff]
  return result


def load_results(results_csv, dataset, model, payoff):
  if not os.path.exists(results_csv):
    raise FileNotFoundError(f"Could not find results file:\n{results_csv}")

  df = pd.read_csv(results_csv)
  subset = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["payoff"] == payoff)].copy()

  if subset.empty:
    raise ValueError(f"No rows found for dataset={dataset!r}, model={model!r}, payoff={payoff!r}.")

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


def print_summary(agreement_summary, fidelity_summary, statistical_tests, target):
  print()
  print("=" * 80)
  print("E4 MASKER AGREEMENT (mean/std + coverage)")
  print("=" * 80)
  print(agreement_summary.to_string(index=False))

  print()
  print("=" * 80)
  print(f"E4 FIDELITY ON THE {'DIRECTIONAL' if target == 'dir' else 'PRESERVATION'} TARGET")
  if target == "pres":
    print("  Sign agreement saturates here: the observed effect is non-negative")
    print("  by definition, so a uniformly positive attribution attains 1.0.")
  print("=" * 80)
  print(fidelity_summary.to_string(index=False))

  print()
  print("=" * 80)
  print("E4 WILCOXON: DELETE VS REPLACEMENT")
  print("=" * 80)
  print(statistical_tests.to_string(index=False))


async def run(model, dataset, payoff, results_csv, output_dir):
  target = PAYOFF_TARGET[payoff]
  wide = load_results(results_csv, dataset, model, payoff)
  wide, n_attempted, n_skipped = await compute_deltas(wide, dataset, model)
  print(f"Samples skipped (feature/token count mismatch): {n_skipped} / {n_attempted + n_skipped}")

  agreement = masker_agreement(wide)
  agreement.to_csv(os.path.join(output_dir, "e4_masker_agreement_per_sample.csv"), index=False)
  agreement_summary = summarize_agreement(agreement, n_attempted)
  agreement_summary.to_csv(os.path.join(output_dir, "e4_masker_agreement_summary.csv"), index=False)

  fidelity = analyze_fidelity(wide, target)
  fidelity.to_csv(os.path.join(output_dir, "e4_fidelity_per_sample.csv"), index=False)
  fidelity_summary = summarize_fidelity(fidelity, n_attempted)
  fidelity_summary["target"] = target
  fidelity_summary.to_csv(os.path.join(output_dir, "e4_fidelity_summary.csv"), index=False)

  statistical_tests = run_statistical_tests(fidelity, model, dataset, payoff)
  statistical_tests.to_csv(os.path.join(output_dir, "e4_wilcoxon_raw.csv"), index=False)

  print_summary(agreement_summary, fidelity_summary, statistical_tests, target)


@click.command()
@click.option("--model_name", type=click.Choice(MODELS), required=True)
@click.option("--dataset_name", type=click.Choice(DATASETS), required=True)
@click.option("--payoff_name", type=click.Choice(PAYOFFS), required=True)
@click.option("--results_csv", type=str, default=DEFAULT_RESULTS_CSV, show_default=True)
def main(model_name, dataset_name, payoff_name, results_csv):
  output_dir = os.path.join(RESULTS_DIR, "e4", f"{dataset_name}_{model_name}_{payoff_name}")
  os.makedirs(output_dir, exist_ok=True)
  asyncio.run(run(model_name, dataset_name, payoff_name, results_csv, output_dir))


if __name__ == "__main__":
  main()
