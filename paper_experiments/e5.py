import json
import os

import click
import numpy as np
import pandas as pd

from ordinalshap.visualization import ShapleyVisualization

CACHE_DIR = "response_cache"

def sign_disagreement(a, b):
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)
  sa, sb = np.sign(a), np.sign(b)
  mask = (sa != 0) & (sb != 0)
  if mask.sum() == 0:
    return np.nan
  return float(np.mean(sa[mask] != sb[mask]))


def response_diversity(dataset, model, masker, sample_id, n_samples=10, sampler_name="permutation"):
  path = os.path.join(CACHE_DIR, f"{dataset}_{model}_{masker}_{sampler_name}_{n_samples}_{sample_id}.json")
  if not os.path.exists(path):
    return 0
  with open(path) as f:
    data = json.load(f)
  return len(set(data.values()))


def find_best_example(results_csv, model, dataset, masker, baseline_payoff, min_response_diversity=3, n_samples=10):

  df = pd.read_csv(results_csv)
  subset = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["masker"] == masker)]

  ordinal = subset[subset["payoff"] == "ordinal"]
  baseline = subset[subset["payoff"] == baseline_payoff]

  if ordinal.empty or baseline.empty:
    raise ValueError(f"No rows found for ordinal or {baseline_payoff!r} under model={model!r}, dataset={dataset!r}, masker={masker!r}.")

  best = None
  n_skipped_degenerate = 0
  n_skipped_no_cache = 0

  for sample_id, group_o in ordinal.groupby("sample_id"):
    group_b = baseline[baseline["sample_id"] == sample_id]
    if len(group_o) != len(group_b) or len(group_o) == 0:
      continue

    diversity = response_diversity(dataset, model, masker, sample_id, n_samples=n_samples)
    if diversity == 0:
      n_skipped_no_cache += 1
      continue
    if diversity < min_response_diversity:
      n_skipped_degenerate += 1
      continue

    merged = group_o.merge(group_b, on="feature", suffixes=("_ord", "_base")).sort_values("feature")

    sd = sign_disagreement(merged["shapley_ord"].values, merged["shapley_base"].values)
    if np.isnan(sd):
      continue

    if best is None or sd > best["sign_disagreement"] or (sd == best["sign_disagreement"] and sample_id < best["sample_id"]):
      best = {
        "sample_id": sample_id,
        "sign_disagreement": sd,
        "response_diversity": diversity,
        "tokens": merged["token_ord"].tolist(),
        "ordinal_values": merged["shapley_ord"].values,
        "baseline_values": merged["shapley_base"].values,
      }

  print(f"Skipped {n_skipped_degenerate} sample(s) with fewer than {min_response_diversity} distinct responses, and {n_skipped_no_cache} sample(s) with no cached responses found.")

  if best is None:
    raise ValueError(f"No comparable, non-degenerate samples found for {model}/{dataset}/{masker}/{baseline_payoff}. Consider lowering min_response_diversity or checking the response_cache path.")

  return best


@click.command()
@click.option("--results_csv", type=str, default="results/results.csv")
@click.option("--model", type=str, default="deepseek")
@click.option("--dataset", type=str, default="sst-5")
@click.option("--masker", type=str, default="delete")
@click.option("--baseline_payoff", type=str, default="tf-idf")
@click.option("--top_k", type=int, default=10)
@click.option("--min_response_diversity", type=int, default=3, help="Minimum number of distinct model responses required across a sample's cached coalitions for it to be eligible for selection.")
@click.option("--n_samples", type=int, default=10, help="Must match the n_samples used when the response_cache files were generated.")
def main(results_csv, model, dataset, masker, baseline_payoff, top_k, min_response_diversity, n_samples):
  RESULTS_DIR = "results"
  output_dir = os.path.join(RESULTS_DIR, "e5")
  os.makedirs(output_dir, exist_ok=True)

  best = find_best_example(
    results_csv, model, dataset, masker, baseline_payoff,
    min_response_diversity=min_response_diversity, n_samples=n_samples,
  )

  print(f"Selected sample_id={best['sample_id']} (sign disagreement = {best['sign_disagreement']:.3f}, response diversity = {best['response_diversity']})")
  print(" ".join(best["tokens"]))
  print()
  print("token, ordinal, " + baseline_payoff)
  for token, o, b in zip(best["tokens"], best["ordinal_values"], best["baseline_values"]):
    print(f"  {token:15s} {o:+.3f}   {b:+.3f}")

  ordinal_vis = ShapleyVisualization(best["tokens"], best["ordinal_values"], label="Ordinal")
  baseline_vis = ShapleyVisualization(best["tokens"], best["baseline_values"], label=baseline_payoff.upper())

  bar_ordinal_path = os.path.join(output_dir, "case_study_bar_ordinal.pdf")
  bar_baseline_path = os.path.join(output_dir, f"case_study_bar_{baseline_payoff}.pdf")

  ordinal_vis.plot_bar_chart(bar_ordinal_path, top_k=top_k)
  baseline_vis.plot_bar_chart(bar_baseline_path, top_k=top_k)


if __name__ == "__main__":
  main()