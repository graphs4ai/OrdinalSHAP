import os

import click
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

RESULTS_DIR = "results"
DEFAULT_RESULTS_CSV = "results/results.csv"

MODELS = ["deepseek", "gemma", "mistral"]
DATASETS = ["sst-5", "yelp-20", "yelp-50"]
MASKER = "delete"

PAYOFF_COLUMN_MAP = {
  "ordinal": "ordinal",
  "tf-idf": "tfidf",
  "minilm": "minilm",
}

PAIRED_TEST_COMPARISONS = [
  ("ordinal_vs_tfidf", "minilm_vs_tfidf"),
  ("ordinal_vs_minilm", "minilm_vs_tfidf"),
]


def spearman(x, y):
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)

  if np.all(x == x[0]) or np.all(y == y[0]):
    return np.nan

  return spearmanr(x, y).statistic


def sign_disagreement(x, y):
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)

  sign_x = np.sign(x)
  sign_y = np.sign(y)

  mask = (sign_x != 0) & (sign_y != 0)
  n_excluded = int((~mask).sum())
  n_total = len(mask)

  if mask.sum() == 0:
    return np.nan, n_excluded, n_total

  return np.mean(sign_x[mask] != sign_y[mask]), n_excluded, n_total


def load_results(results_csv, dataset, model, masker):
  if not os.path.exists(results_csv):
    raise FileNotFoundError(f"Could not find results file:\n{results_csv}")

  df = pd.read_csv(results_csv)
  subset = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["masker"] == masker)]
  frames = {}
  for payoff_value, internal_name in PAYOFF_COLUMN_MAP.items():
    part = subset[subset["payoff"] == payoff_value]
    frames[internal_name] = part[["sample_id", "feature", "token", "shapley"]]

  return frames["ordinal"], frames["tfidf"], frames["minilm"]


def merge_results(ordinal, tfidf, minilm):
  ordinal = ordinal[["sample_id", "feature", "token", "shapley"]].rename(columns={"shapley": "ordinal"})
  tfidf = tfidf[["sample_id", "feature", "shapley"]].rename(columns={"shapley": "tfidf"})
  minilm = minilm[["sample_id", "feature", "shapley"]].rename(columns={"shapley": "minilm"})

  df = ordinal.merge(tfidf, on=["sample_id", "feature"], how="inner")
  df = df.merge(minilm, on=["sample_id", "feature"], how="inner")

  return df


def analyze_samples(df):
  results = []
  for sample_id, group in df.groupby("sample_id"):
    ordinal = group["ordinal"].values
    tfidf = group["tfidf"].values
    minilm = group["minilm"].values

    for comparison, (a, b) in [("ordinal_vs_tfidf", (ordinal, tfidf)), ("ordinal_vs_minilm", (ordinal, minilm)), ("minilm_vs_tfidf", (minilm, tfidf))]:
      sd, n_excluded, n_total = sign_disagreement(a, b)
      results.append({
        "sample_id": sample_id,
        "comparison": comparison,
        "spearman": spearman(a, b),
        "sign_disagreement": sd,
        "sign_disagreement_n_excluded": n_excluded,
        "sign_disagreement_n_total": n_total,
        "n_tokens": len(group),
      })

  return pd.DataFrame(results)


def create_summary(per_sample):
  rows = []

  for comparison, group in per_sample.groupby("comparison"):
    spearman_vals = group["spearman"].dropna()
    rows.append({
      "comparison": comparison,
      "metric": "spearman",
      "mean": spearman_vals.mean(),
      "std": spearman_vals.std(),
      "n_valid": len(spearman_vals),
      "n_total": len(group),
      "coverage": len(spearman_vals) / len(group) if len(group) else np.nan,
    })

    sd_vals = group["sign_disagreement"].dropna()
    n_excluded_tokens = group["sign_disagreement_n_excluded"].sum()
    n_total_tokens = group["sign_disagreement_n_total"].sum()
    rows.append({
      "comparison": comparison,
      "metric": "sign_disagreement",
      "mean": sd_vals.mean(),
      "std": sd_vals.std(),
      "n_valid": len(sd_vals),
      "n_total": len(group),
      "coverage": len(sd_vals) / len(group) if len(group) else np.nan,
      "token_exclusion_rate": n_excluded_tokens / n_total_tokens if n_total_tokens else np.nan,
    })

  return pd.DataFrame(rows)


def rank_biserial_effect_size(diff):
  diff = diff[diff != 0]
  n = len(diff)
  if n == 0:
    return np.nan, 0

  ranks = pd.Series(np.abs(diff)).rank().values
  w_pos = ranks[diff > 0].sum()
  w_neg = ranks[diff < 0].sum()

  return (w_pos - w_neg) / (w_pos + w_neg), n


def paired_wilcoxon_test(per_sample, metric, comp_a, comp_b):
  a = per_sample[per_sample["comparison"] == comp_a].set_index("sample_id")[metric]
  b = per_sample[per_sample["comparison"] == comp_b].set_index("sample_id")[metric]

  common = a.index.intersection(b.index)
  a = a.loc[common].dropna()
  b = b.loc[common].dropna()
  common = a.index.intersection(b.index)
  a, b = a.loc[common], b.loc[common]

  if len(common) < 2:
    return {
      "metric": metric, 
      "comparison_a": comp_a, 
      "comparison_b": comp_b,
      "statistic": np.nan, 
      "p_value": np.nan, 
      "effect_size": np.nan, 
      "n": len(common),
    }

  diff = (a - b).values
  try:
    stat, p = wilcoxon(a, b)
  except ValueError:
    stat, p = np.nan, np.nan

  effect, n_nonzero = rank_biserial_effect_size(diff)

  return {
    "metric": metric,
    "comparison_a": comp_a,
    "comparison_b": comp_b,
    "statistic": stat,
    "p_value": p,
    "effect_size": effect,
    "n": len(common),
  }


def run_statistical_tests(per_sample):
  wilcoxon_rows = []
  for metric in ["spearman", "sign_disagreement"]:
    for comp_a, comp_b in PAIRED_TEST_COMPARISONS:
      wilcoxon_rows.append(paired_wilcoxon_test(per_sample, metric, comp_a, comp_b))
  return pd.DataFrame(wilcoxon_rows)


def print_summary(summary, wilcoxon_df):
  print()
  print("=" * 80)
  print("E2 SUMMARY (mean/std + coverage)")
  print("=" * 80)

  for comparison in summary["comparison"].unique():
    print()
    print(comparison)
    subset = summary[summary["comparison"] == comparison]
    for _, row in subset.iterrows():
      extra = ""
      if row["metric"] == "sign_disagreement" and not np.isnan(row.get("token_exclusion_rate", np.nan)):
        extra = f" token_exclusion_rate={row['token_exclusion_rate']:.3f}"
      print(f" {row['metric']:20s} mean={row['mean']:.3f} std={row['std']:.3f} coverage={row['coverage']:.3f} ({row['n_valid']}/{row['n_total']}){extra}")

  print()
  print("=" * 80)
  print("WILCOXON PAIRED TESTS")
  print("=" * 80)
  print(wilcoxon_df.to_string(index=False))


@click.command()
@click.option("--model_name", type=click.Choice(MODELS), required=True)
@click.option("--dataset_name", type=click.Choice(DATASETS), required=True)
@click.option("--results_csv", type=str, default=DEFAULT_RESULTS_CSV, show_default=True)
def main(model_name, dataset_name, results_csv):
  output_dir = os.path.join(RESULTS_DIR, "e2", f"{dataset_name}_{model_name}_{MASKER}")
  os.makedirs(output_dir, exist_ok=True)

  ordinal, tfidf, minilm = load_results(results_csv, dataset_name, model_name, MASKER)

  df = merge_results(ordinal, tfidf, minilm)
  per_sample = analyze_samples(df)
  per_sample.to_csv(os.path.join(output_dir, "e2_per_sample.csv"), index=False)

  summary = create_summary(per_sample)
  summary.to_csv(os.path.join(output_dir, "e2_summary.csv"), index=False)

  wilcoxon_df = run_statistical_tests(per_sample)
  wilcoxon_df["model"] = model_name
  wilcoxon_df["dataset"] = dataset_name
  wilcoxon_df["masker"] = MASKER
  wilcoxon_df.to_csv(os.path.join(output_dir, "e2_wilcoxon_raw.csv"), index=False)

  print_summary(summary, wilcoxon_df)


if __name__ == "__main__":
  main()