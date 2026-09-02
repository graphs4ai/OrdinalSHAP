import os

import click
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

RESULTS_DIR = "results"
DEFAULT_RESULTS_CSV = "results/results.csv"

MODELS = ["deepseek", "gemma", "mistral"]
DATASETS = ["sst5-20", "yelp-20", "yelp-50"]
MASKER = "delete"

PAYOFF_COLUMN_MAP = {
  "ordinal-directional": "ordinal",
  "ordinal-preservation": "ordpres",
  "tf-idf": "tfidf",
  "minilm": "minilm",
}

COMPARISONS = [
  ("ordinal_vs_tfidf", ("ordinal", "tfidf")),
  ("ordinal_vs_minilm", ("ordinal", "minilm")),
  ("ordinal_vs_ordpres", ("ordinal", "ordpres")),
  ("ordpres_vs_tfidf", ("ordpres", "tfidf")),
  ("ordpres_vs_minilm", ("ordpres", "minilm")),
  ("minilm_vs_tfidf", ("minilm", "tfidf")),
]

PAIRED_TEST_COMPARISONS = [
  ("ordinal_vs_tfidf", "minilm_vs_tfidf"),
  ("ordinal_vs_minilm", "minilm_vs_tfidf"),
  ("ordpres_vs_tfidf", "minilm_vs_tfidf"),
  ("ordpres_vs_minilm", "minilm_vs_tfidf"),
]


EMPTY_COALITION_RANK = 2
EFFICIENCY_TOL = 1e-6


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


def ternary_disagreement(x, y):
  sign_x = np.sign(np.asarray(x, dtype=float))
  sign_y = np.sign(np.asarray(y, dtype=float))
  if len(sign_x) == 0:
    return np.nan
  return float(np.mean(sign_x != sign_y))


def load_results(results_csv, dataset, model, masker):
  if not os.path.exists(results_csv):
    raise FileNotFoundError(f"Could not find results file:\n{results_csv}")

  df = pd.read_csv(results_csv).drop_duplicates(subset=["dataset","model","masker","payoff","sample_id","feature"], keep="last")
  subset = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["masker"] == masker)]

  if subset.empty:
    raise ValueError(f"No rows for {dataset}/{model}/{masker} in {results_csv}")

  frames = {}
  for payoff_value, internal_name in PAYOFF_COLUMN_MAP.items():
    part = subset[subset["payoff"] == payoff_value]
    if part.empty:
      available = sorted(subset["payoff"].unique())
      raise ValueError(f"No rows for payoff={payoff_value!r} in {dataset}/{model}/{masker}. Available: {available}")
    frames[internal_name] = part[["sample_id", "feature", "token", "shapley"]]

  return frames


def merge_results(frames):
  names = list(PAYOFF_COLUMN_MAP.values())

  base = frames[names[0]][["sample_id", "feature", "token", "shapley"]]
  base = base.rename(columns={"shapley": names[0]})
  n_start = len(base)

  df = base
  for name in names[1:]:
    part = frames[name][["sample_id", "feature", "shapley"]].rename(columns={"shapley": name})
    df = df.merge(part, on=["sample_id", "feature"], how="inner")

  if len(df) != n_start:
    print(f"[warn] merge dropped {n_start - len(df)} of {n_start} rows; payoffs have different coverage")

  return df


def analyze_samples(df, check_efficiency=True):
  results = []
  efficiency_failures = []

  for sample_id, group in df.groupby("sample_id"):
    values = {name: group[name].values for name in PAYOFF_COLUMN_MAP.values()}

    phi_sum = float(group["ordinal"].sum())
    o0 = int(round(phi_sum)) + EMPTY_COALITION_RANK
    if check_efficiency and abs(phi_sum - round(phi_sum)) > EFFICIENCY_TOL:
      efficiency_failures.append((sample_id, phi_sum))

    for comparison, (a_name, b_name) in COMPARISONS:
      a, b = values[a_name], values[b_name]
      sd, n_excluded, n_total = sign_disagreement(a, b)
      results.append({
        "sample_id": sample_id,
        "comparison": comparison,
        "o0": o0,
        "phi_dir_sum": phi_sum,
        "spearman": spearman(a, b),
        "sign_disagreement": sd,
        "ternary_disagreement": ternary_disagreement(a, b),
        "sign_disagreement_n_excluded": n_excluded,
        "sign_disagreement_n_total": n_total,
        "n_tokens": len(group),
      })

  if efficiency_failures:
    print(f"[warn] {len(efficiency_failures)} samples with non-integer sum(phi_dir):")
    for sample_id, s in efficiency_failures[:10]:
      print(f" sample {sample_id}: {s:.6f}")
    print(" Expected o_0 - 2 exactly. Check the estimator or build_payoff().")

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
      "token_exclusion_rate": np.nan,
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

    tern_vals = group["ternary_disagreement"].dropna()
    rows.append({
      "comparison": comparison,
      "metric": "ternary_disagreement",
      "mean": tern_vals.mean(),
      "std": tern_vals.std(),
      "n_valid": len(tern_vals),
      "n_total": len(group),
      "coverage": len(tern_vals) / len(group) if len(group) else np.nan,
      "token_exclusion_rate": 0.0,
    })

  return pd.DataFrame(rows)


def create_o0_stratification(per_sample):
  rows = []
  for (comparison, o0), group in per_sample.groupby(["comparison", "o0"]):
    sd_vals = group["sign_disagreement"].dropna()
    rho_vals = group["spearman"].dropna()
    rows.append({
      "comparison": comparison,
      "o0": o0,
      "n_samples": len(group),
      "sd_mean": sd_vals.mean(),
      "sd_std": sd_vals.std(),
      "sd_n": len(sd_vals),
      "spearman_mean": rho_vals.mean(),
      "spearman_n": len(rho_vals),
    })
  return pd.DataFrame(rows).sort_values(["comparison", "o0"])


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

  base = {
    "metric": metric,
    "comparison_a": comp_a,
    "comparison_b": comp_b,
    "statistic": np.nan,
    "p_value": np.nan,
    "effect_size": np.nan,
    "n": len(common),
    "n_nonzero": 0,
    "min_attainable_p": np.nan,
  }

  if len(common) < 2:
    return base

  diff = (a - b).values
  effect, n_nonzero = rank_biserial_effect_size(diff)

  try:
    stat, p = wilcoxon(a, b, zero_method="wilcox")
  except ValueError:
    stat, p = np.nan, np.nan

  base.update({
    "statistic": stat,
    "p_value": p,
    "effect_size": effect,
    "n_nonzero": n_nonzero,
    "min_attainable_p": 2.0 / (2 ** n_nonzero) if 0 < n_nonzero < 60 else 0.0,
  })
  return base


def run_statistical_tests(per_sample):
  wilcoxon_rows = []
  for metric in ["spearman", "sign_disagreement"]:
    for comp_a, comp_b in PAIRED_TEST_COMPARISONS:
      wilcoxon_rows.append(paired_wilcoxon_test(per_sample, metric, comp_a, comp_b))
  return pd.DataFrame(wilcoxon_rows)


def print_summary(summary, wilcoxon_df, strat, per_sample):
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
      ter = row.get("token_exclusion_rate", np.nan)
      if row["metric"] == "sign_disagreement" and not np.isnan(ter):
        extra = f" token_exclusion_rate={ter:.3f}"
      print(f"  {row['metric']:22s} mean={row['mean']:.3f} std={row['std']:.3f} coverage={row['coverage']:.3f} ({row['n_valid']}/{row['n_total']}){extra}")

  print()
  print("=" * 80)
  print("SIGN DISAGREEMENT STRATIFIED BY o_0")
  print("=" * 80)
  o0_counts = per_sample.drop_duplicates("sample_id")["o0"].value_counts().sort_index()
  print("\no_0 distribution:", dict(o0_counts))
  for comparison in strat["comparison"].unique():
    print(f"\n{comparison}")
    sub = strat[strat["comparison"] == comparison]
    print(f"  {'o_0':>4} {'n':>4} {'SD mean':>9} {'SD std':>8} {'rho mean':>9}")
    for _, r in sub.iterrows():
      print(f"  {int(r['o0']):>4} {int(r['n_samples']):>4} {r['sd_mean']:>9.3f} {r['sd_std']:>8.3f} {r['spearman_mean']:>9.3f}")

  print()
  print("=" * 80)
  print("WILCOXON PAIRED TESTS")
  print("=" * 80)
  print(wilcoxon_df.to_string(index=False))


@click.command()
@click.option("--model_name", type=click.Choice(MODELS), required=True)
@click.option("--dataset_name", type=click.Choice(DATASETS), required=True)
@click.option("--masker_name", type=click.Choice(["delete", "replacement"]), default=MASKER, show_default=True)
@click.option("--results_csv", type=str, default=DEFAULT_RESULTS_CSV, show_default=True)
def main(model_name, dataset_name, masker_name, results_csv):
  output_dir = os.path.join(RESULTS_DIR, "e2", f"{dataset_name}_{model_name}_{masker_name}")
  os.makedirs(output_dir, exist_ok=True)

  frames = load_results(results_csv, dataset_name, model_name, masker_name)
  df = merge_results(frames)

  per_sample = analyze_samples(df)
  for col, val in [("model", model_name), ("dataset", dataset_name), ("masker", masker_name)]:
    per_sample[col] = val
  per_sample.to_csv(os.path.join(output_dir, "e2_per_sample.csv"), index=False)

  summary = create_summary(per_sample)
  for col, val in [("model", model_name), ("dataset", dataset_name), ("masker", masker_name)]:
    summary[col] = val
  summary.to_csv(os.path.join(output_dir, "e2_summary.csv"), index=False)

  strat = create_o0_stratification(per_sample)
  for col, val in [("model", model_name), ("dataset", dataset_name), ("masker", masker_name)]:
    strat[col] = val
  strat.to_csv(os.path.join(output_dir, "e2_o0_stratification.csv"), index=False)

  wilcoxon_df = run_statistical_tests(per_sample)
  for col, val in [("model", model_name), ("dataset", dataset_name), ("masker", masker_name)]:
    wilcoxon_df[col] = val
  wilcoxon_df.to_csv(os.path.join(output_dir, "e2_wilcoxon_raw.csv"), index=False)

  print_summary(summary, wilcoxon_df, strat, per_sample)


if __name__ == "__main__":
  main()
