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
MASKER = "delete"
PAYOFF_METHODS = ["ordinal-directional", "ordinal-preservation", "tf-idf", "minilm", "bm25", "google-embeddings-2", "bge", "gte", "e5"]
CHANCE_METHODS = ["random", "constant-positive", "constant-negative"]
DEFAULT_METHODS = PAYOFF_METHODS + CHANCE_METHODS

REFERENCE = {
  "dir": "ordinal-directional",
  "pres": "ordinal-preservation",
}

MODEL_NAMES = {
  "deepseek": "deepseek-ai/DeepSeek-V4-Flash-0731",
  "gemma": "google/gemma-4-26B-A4B-it",
  "mistral": "mistralai/Mistral-Small-24B-Instruct-2501",
}

API_KEY = os.environ["DEEPINFRA_API_KEY"]
BASE_URL = "https://api.deepinfra.com/v1/openai"

K_MAX_FRACTION = 0.30
CHANCE_SEED = 0

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

YELP_MAPPING = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}

DATASET_CONFIGS = {
  "sst": {"system_prompt": SST5_SYSTEM_PROMPT, "mapping": SST5_MAPPING},
  "yelp": {"system_prompt": YELP_SYSTEM_PROMPT, "mapping": YELP_MAPPING},
}


def dataset_config(dataset_name):
  return DATASET_CONFIGS[dataset_name.split("-")[0]]


class FallbackCounter:
  def __init__(self):
    self.total_calls = 0
    self.fallback_calls = 0

  def rate(self):
    return self.fallback_calls / self.total_calls if self.total_calls else float("nan")


def ordinal_value(response, mapping, counter=None):
  if counter is not None:
    counter.total_calls += 1
  norm = response.strip()
  if norm in mapping:
    return float(mapping[norm])
  if counter is not None:
    counter.fallback_calls += 1
  values = list(mapping.values())
  return (max(values) + min(values)) / 2.0


def get_k_values(n_tokens):
  k_max = max(1, round(K_MAX_FRACTION * n_tokens))
  return list(range(1, k_max + 1))


def load_shap_results(results_csv, dataset, model, masker, methods):
  if not os.path.exists(results_csv):
    raise FileNotFoundError(f"Could not find results file:\n{results_csv}")

  df = pd.read_csv(results_csv)
  df = df.drop_duplicates(subset=["dataset", "model", "masker", "payoff", "sample_id", "feature"], keep="last")
  subset = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["masker"] == masker)]

  if subset.empty:
    raise ValueError(f"No rows for dataset={dataset!r}, model={model!r}, masker={masker!r}.")

  data = {}
  for method in methods:
    if method in CHANCE_METHODS:
      continue
    part = subset[subset["payoff"] == method]
    if part.empty:
      raise ValueError(f"Method {method!r} not found for {dataset}/{model}/{masker}. Available payoffs: {sorted(subset['payoff'].unique())}")
    data[method] = part

  return data


def magnitude_ranking(df, sample_id):
  sample = df[df["sample_id"] == sample_id].copy()
  sample["abs_shapley"] = sample["shapley"].abs()
  return sample.sort_values("abs_shapley", ascending=False).reset_index(drop=True)


def chance_ranking(method, n_tokens, sample_id, dataset, model):
  seed = abs(hash((CHANCE_SEED, dataset, model, sample_id, method))) % (2**32)
  rng = np.random.default_rng(seed)

  if method == "random":
    order = rng.permutation(n_tokens)
    values = rng.normal(size=n_tokens)
  elif method == "constant-positive":
    order = np.arange(n_tokens)
    values = np.ones(n_tokens)
  elif method == "constant-negative":
    order = np.arange(n_tokens)
    values = -np.ones(n_tokens)
  else:
    raise ValueError(f"Unknown chance method: {method!r}")

  return pd.DataFrame({"feature": order, "shapley": values})


def build_deletion_text(tokens, ordered_features, k, original_text):
  if k == 0:
    return original_text
  removed = set(ordered_features[:k])
  remaining = [token for i, token in enumerate(tokens) if i not in removed]
  return " ".join(remaining)


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


def rank_correlation(delta_real, delta_hat):
  delta_real = np.asarray(delta_real, dtype=float)
  delta_hat = np.asarray(delta_hat, dtype=float)

  if len(delta_real) < 2:
    return np.nan
  if np.all(delta_real == delta_real[0]):
    return np.nan
  if np.all(delta_hat == delta_hat[0]):
    return np.nan

  return spearmanr(delta_real, delta_hat).statistic


def sign_agreement(delta_real, delta_hat):
  delta_real = np.asarray(delta_real, dtype=float)
  delta_hat = np.asarray(delta_hat, dtype=float)

  mask = delta_real != 0
  if mask.sum() == 0:
    return np.nan

  return float(np.mean(np.sign(delta_real[mask]) == np.sign(delta_hat[mask])))


def mean_absolute_error(delta_real, delta_hat):
  delta_real = np.asarray(delta_real, dtype=float)
  delta_hat = np.asarray(delta_hat, dtype=float)
  if len(delta_real) == 0:
    return np.nan
  return float(np.mean(np.abs(delta_real - delta_hat)))


async def analyze(model_name, dataset_name, masker_name, methods, results_csv):
  config = dataset_config(dataset_name)
  system_prompt = config["system_prompt"]
  mapping = config["mapping"]
  n_classes = len(mapping)
  span = float(max(mapping.values()) - min(mapping.values()))
  model_full_name = MODEL_NAMES[model_name]

  shap_data = load_shap_results(results_csv, dataset_name, model_name, masker_name, methods)
  data = pd.read_csv(f"data/{dataset_name}.csv")

  model = InstanceModel(
    model_name=model_full_name,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.0,
    max_tokens=5,
    max_concurrency=8,
    requests_per_minute=50,
  )

  splitter = WordSplitter()

  output_dir = os.path.join(RESULTS_DIR, "e3", f"{dataset_name}_{model_name}_{masker_name}")
  os.makedirs(output_dir, exist_ok=True)
  output_path = os.path.join(output_dir, "e3_fidelity_results.csv")

  fallback_counter = FallbackCounter()
  n_attempted = {method: 0 for method in methods}
  n_ranking_incomplete = {method: 0 for method in methods}
  efficiency_failures = []

  results = []
  removed = 0
  n_total_samples = len(data)

  for sample_id, row in tqdm(data.iterrows(), total=len(data)):
    text = row["text"]
    tokens = splitter.split(text)
    n_tokens = len(tokens)
    k_values = get_k_values(n_tokens)

    if n_tokens < 3:
      removed += 1
      continue

    text_cache, cpath = load_text_cache(dataset_name, model_name, sample_id, text)

    if text not in text_cache:
      text_cache[text] = await model.generate(user_prompt=text, system_prompt=system_prompt)

    full_response = text_cache[text]
    full_ordinal = ordinal_value(full_response, mapping, fallback_counter)

    method_rankings = {}
    for method in methods:
      if method in CHANCE_METHODS:
        method_rankings[method] = chance_ranking(method, n_tokens, sample_id, dataset_name, model_name)
        n_attempted[method] += 1
        continue

      ranking = magnitude_ranking(shap_data[method], sample_id)
      if len(ranking) != n_tokens:
        n_ranking_incomplete[method] += 1
        continue
      method_rankings[method] = ranking
      n_attempted[method] += 1

    if "ordinal-directional" in method_rankings:
      phi_sum = float(method_rankings["ordinal-directional"]["shapley"].sum())
      if abs(phi_sum - round(phi_sum)) > 1e-6:
        efficiency_failures.append((sample_id, phi_sum))

    cell_texts = {}
    for method, ranking in method_rankings.items():
      ordered_features = ranking["feature"].tolist()
      cell_texts[method] = [build_deletion_text(tokens, ordered_features, k, text) for k in k_values]

    all_texts = set()
    for texts in cell_texts.values():
      all_texts.update(texts)

    to_query = [t for t in all_texts if t not in text_cache]
    if to_query:
      responses = await model.batch_generate(prompts=to_query, system_prompt=system_prompt)
      for t, r in zip(to_query, responses):
        text_cache[t] = r

    save_text_cache(cpath, text, text_cache)

    for method, ranking in method_rankings.items():
      texts = cell_texts[method]

      delta_real_dir = []
      delta_real_pres = []
      delta_hat = []

      for k, deletion_text in zip(k_values, texts):
        coalition_ordinal = ordinal_value(text_cache[deletion_text], mapping, fallback_counter)
        delta_real_dir.append(full_ordinal - coalition_ordinal)
        delta_real_pres.append(abs(full_ordinal - coalition_ordinal) / span)
        delta_hat.append(ranking["shapley"].iloc[:k].sum())

      delta_real_dir = np.asarray(delta_real_dir, dtype=float)
      delta_real_pres = np.asarray(delta_real_pres, dtype=float)
      delta_hat = np.asarray(delta_hat, dtype=float)

      results.append({
        "dataset": dataset_name,
        "model": model_name,
        "masker": masker_name,
        "method": method,
        "sample_id": sample_id,
        "n_tokens": n_tokens,
        "o0": int(round(full_ordinal)),
        "sa_dir": sign_agreement(delta_real_dir, delta_hat),
        "rc_dir": rank_correlation(delta_real_dir, delta_hat),
        "mae_dir": mean_absolute_error(delta_real_dir, delta_hat),
        "sa_pres": sign_agreement(delta_real_pres, delta_hat),
        "rc_pres": rank_correlation(delta_real_pres, delta_hat),
        "mae_pres": mean_absolute_error(delta_real_pres, delta_hat),
      })

    pd.DataFrame(results).to_csv(output_path, index=False)

  print(f"Samples removed (n_tokens < 3): {removed}")
  print(f"Ordinal-value fallback rate: {fallback_counter.rate():.4f} ({fallback_counter.fallback_calls}/{fallback_counter.total_calls})")
  if efficiency_failures:
    print(f"[warn] {len(efficiency_failures)} samples with non-integer sum(phi_dir):")
    for sample_id, s in efficiency_failures[:10]:
      print(f" sample {sample_id}: {s:.6f}")
  for method in methods:
    if n_ranking_incomplete[method]:
      print(f"[warn] {method}: {n_ranking_incomplete[method]} samples with an incomplete ranking, excluded")

  diagnostics_rows = [
    {"diagnostic": "n_total_samples", "value": n_total_samples},
    {"diagnostic": "n_removed_short_text", "value": removed},
    {"diagnostic": "n_classes", "value": n_classes},
    {"diagnostic": "ordinal_value_fallback_rate", "value": fallback_counter.rate()},
    {"diagnostic": "ordinal_value_fallback_count", "value": fallback_counter.fallback_calls},
    {"diagnostic": "ordinal_value_total_calls", "value": fallback_counter.total_calls},
    {"diagnostic": "n_efficiency_failures", "value": len(efficiency_failures)},
  ]
  for method in methods:
    diagnostics_rows.append({"diagnostic": f"n_attempted__{method}", "value": n_attempted[method]})
    diagnostics_rows.append({"diagnostic": f"n_ranking_incomplete__{method}", "value": n_ranking_incomplete[method]})
  diagnostics = pd.DataFrame(diagnostics_rows)
  for col, val in [("dataset", dataset_name), ("model", model_name), ("masker", masker_name)]:
    diagnostics[col] = val
  diagnostics.to_csv(os.path.join(output_dir, "e3_diagnostics.csv"), index=False)

  return pd.DataFrame(results), output_dir, n_attempted


def rank_biserial_effect_size(diff):
  diff = np.asarray(diff, dtype=float)
  diff = diff[diff != 0]
  if len(diff) == 0:
    return np.nan, 0
  ranks = pd.Series(np.abs(diff)).rank().values
  w_pos = ranks[diff > 0].sum()
  w_neg = ranks[diff < 0].sum()
  return (w_pos - w_neg) / (w_pos + w_neg), len(diff)


def wilcoxon_vs_reference(per_sample, metric, reference, methods):
  ref = per_sample[per_sample["method"] == reference].set_index("sample_id")[metric]
  rows = []

  for method in methods:
    if method == reference:
      continue
    other = per_sample[per_sample["method"] == method].set_index("sample_id")[metric]
    common = ref.index.intersection(other.index)

    a = ref.loc[common].dropna()
    b = other.loc[common].dropna()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    row = {
      "metric": metric,
      "reference": reference,
      "comparison": f"{reference}_vs_{method}",
      "method": method,
      "statistic": np.nan,
      "p_value": np.nan,
      "effect_size": np.nan,
      "n": len(common),
      "n_nonzero": 0,
      "min_attainable_p": np.nan,
    }

    if len(common) < 2:
      rows.append(row)
      continue

    diff = a.values - b.values
    effect, n_nonzero = rank_biserial_effect_size(diff)

    try:
      statistic, p_value = wilcoxon(a.values, b.values, zero_method="wilcox")
    except ValueError:
      statistic, p_value = np.nan, np.nan

    row.update({
      "statistic": statistic,
      "p_value": p_value,
      "effect_size": effect,
      "n_nonzero": n_nonzero,
      "min_attainable_p": 2.0 / (2 ** n_nonzero) if 0 < n_nonzero < 60 else 0.0,
    })
    rows.append(row)

  return pd.DataFrame(rows)


def summarize(results_df, methods, n_attempted, output_dir):
  metrics = ["sa_dir", "rc_dir", "mae_dir", "sa_pres", "rc_pres", "mae_pres"]
  rows = []

  for method in methods:
    subset = results_df[results_df["method"] == method]
    denom = n_attempted.get(method, np.nan)
    for metric in metrics:
      values = subset[metric].dropna()
      rows.append({
        "method": method,
        "metric": metric,
        "mean": values.mean(),
        "std": values.std(),
        "n": len(values),
        "n_attempted": denom,
        "coverage": (len(values) / denom) if denom else np.nan,
      })

  summary = pd.DataFrame(rows)
  summary.to_csv(os.path.join(output_dir, "e3_fidelity_summary.csv"), index=False)
  return summary


def stratify_by_o0(results_df, output_dir):
  rows = []
  for (method, o0), group in results_df.groupby(["method", "o0"]):
    rows.append({
      "method": method,
      "o0": o0,
      "n": len(group),
      "sa_dir_mean": group["sa_dir"].dropna().mean(),
      "rc_dir_mean": group["rc_dir"].dropna().mean(),
      "sa_pres_mean": group["sa_pres"].dropna().mean(),
      "rc_pres_mean": group["rc_pres"].dropna().mean(),
    })
  strat = pd.DataFrame(rows).sort_values(["method", "o0"])
  strat.to_csv(os.path.join(output_dir, "e3_o0_stratification.csv"), index=False)
  return strat


def run_statistics(results_df, methods, output_dir, model_name, dataset_name, masker_name):
  target_metrics = {"dir": ["sa_dir", "rc_dir"], "pres": ["sa_pres", "rc_pres"]}

  frames = []
  for target, metrics in target_metrics.items():
    reference = REFERENCE[target]
    for metric in metrics:
      result = wilcoxon_vs_reference(results_df, metric, reference, methods)
      if not result.empty:
        result["target"] = target
        frames.append(result)

  wilcoxon_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
  if len(wilcoxon_df):
    for col, val in [("model", model_name), ("dataset", dataset_name), ("masker", masker_name)]:
      wilcoxon_df[col] = val

  wilcoxon_df.to_csv(os.path.join(output_dir, "e3_wilcoxon_raw.csv"), index=False)
  return wilcoxon_df


def print_results(summary, wilcoxon_df, strat):
  print()
  print("=" * 80)
  print("E3 FIDELITY SUMMARY")
  print("=" * 80)
  print(summary.to_string(index=False))

  print()
  print("=" * 80)
  print("FIDELITY STRATIFIED BY o_0")
  print("=" * 80)
  print(strat.to_string(index=False))

  print()
  print("=" * 80)
  print("PAIRED WILCOXON AGAINST THE ALIGNED PAYOFF")
  print("=" * 80)
  if len(wilcoxon_df):
    print(wilcoxon_df.to_string(index=False))
  else:
    print("No Wilcoxon results.")


@click.command()
@click.option("--model_name", type=click.Choice(MODELS), required=True)
@click.option("--dataset_name", type=click.Choice(DATASETS), required=True)
@click.option("--masker_name", type=click.Choice(["delete", "replacement"]), default=MASKER, show_default=True)
@click.option("--results_csv", type=str, default=DEFAULT_RESULTS_CSV, show_default=True)
@click.option("--methods", type=str, default=",".join(DEFAULT_METHODS), show_default=True)
def main(model_name, dataset_name, masker_name, results_csv, methods):
  methods = methods.split(",")

  results_df, output_dir, n_attempted = asyncio.run(analyze(model_name, dataset_name, masker_name, methods, results_csv))

  summary = summarize(results_df, methods, n_attempted, output_dir)
  strat = stratify_by_o0(results_df, output_dir)
  wilcoxon_df = run_statistics(results_df, methods, output_dir, model_name, dataset_name, masker_name)

  print_results(summary, wilcoxon_df, strat)


if __name__ == "__main__":
  main()
