import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
plt.rcParams.update({"font.size": 14})

RESULTS_DIR = "results"
DEFAULT_RESULTS_CSV = "results/results.csv"

MODELS = ["deepseek", "gemma", "mistral"]
DATASETS = ["sst-5", "yelp-20", "yelp-50"]
MASKER = "delete"

ALL_PAYOFFS = ["ordinal", "tf-idf", "minilm", "bge", "gte", "e5", "google-embeddings-2", "bm25"]


def spearman(x, y):
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  if np.all(x == x[0]) or np.all(y == y[0]):
    return np.nan
  return spearmanr(x, y).statistic


def load_wide(results_csv, dataset, model, masker, payoffs):
  if not os.path.exists(results_csv):
    raise FileNotFoundError(f"Could not find results file:\n{results_csv}")

  df = pd.read_csv(results_csv)
  subset = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["masker"] == masker)]

  frames = []
  for payoff in payoffs:
    part = subset[subset["payoff"] == payoff][["sample_id", "feature", "shapley"]]
    part = part.rename(columns={"shapley": payoff})
    frames.append(part)

  wide = frames[0]
  for part in frames[1:]:
    wide = wide.merge(part, on=["sample_id", "feature"], how="inner")

  return wide


def pair_spearman_values(wide, a, b):
  vals = []
  n_total = 0
  for _, group in wide.groupby("sample_id"):
    n_total += 1
    v = spearman(group[a].values, group[b].values)
    if not np.isnan(v):
      vals.append(v)
  return np.array(vals), n_total


def pair_sign_disagreement(wide, a, b):
  per_sample_vals = []
  n_total = 0
  for _, group in wide.groupby("sample_id"):
    n_total += 1
    sx = np.sign(group[a].values)
    sy = np.sign(group[b].values)
    mask = (sx != 0) & (sy != 0)
    if mask.sum() > 0:
      per_sample_vals.append(np.mean(sx[mask] != sy[mask]))

  return np.array(per_sample_vals), n_total


def build_matrices(wide, payoffs, metric):
  n = len(payoffs)
  mean_matrix = np.full((n, n), np.nan)
  coverage_matrix = np.full((n, n), np.nan)

  for i, a in enumerate(payoffs):
    for j, b in enumerate(payoffs):
      if i == j:
        mean_matrix[i, j] = 1.0 if metric == "spearman" else 0.0
        coverage_matrix[i, j] = 1.0
        continue
      if i > j:
        continue

      if metric == "spearman":
        vals, n_total = pair_spearman_values(wide, a, b)
      elif metric == "sign_disagreement":
        vals, n_total = pair_sign_disagreement(wide, a, b)

      mean_val = vals.mean() if len(vals) else np.nan
      coverage = len(vals) / n_total if n_total else np.nan

      mean_matrix[i, j] = mean_val
      mean_matrix[j, i] = mean_val
      coverage_matrix[i, j] = coverage
      coverage_matrix[j, i] = coverage

  return mean_matrix, coverage_matrix


def plot_matrix(mean_matrix, payoffs, metric_name, output_path, coverage_matrix=None, coverage_threshold=0.5):
  n = len(payoffs)
  fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 2))

  if metric_name == "spearman":
    cmap, vmin, vmax = "RdBu_r", -1, 1
  else:
    cmap, vmin, vmax = "RdBu_r", 0, 1

  im = ax.imshow(mean_matrix, cmap=cmap, vmin=vmin, vmax=vmax)

  ax.set_xticks(range(n))
  ax.set_yticks(range(n))
  ax.set_xticklabels(payoffs, rotation=45, ha="right")
  ax.set_yticklabels(payoffs)

  mid = (vmax + vmin) / 2
  span = (vmax - vmin) / 4

  for i in range(n):
    for j in range(n):
      val = mean_matrix[i, j]
      if np.isnan(val):
        continue
      flag = ""
      if coverage_matrix is not None:
        cov = coverage_matrix[i, j]
        if not np.isnan(cov) and cov < coverage_threshold:
          flag = "*"
      text = f"{val:.2f}{flag}"
      color = "white" if abs(val - mid) > span else "black"
      ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

  fig.colorbar(im, ax=ax, shrink=0.8)
  fig.tight_layout()
  fig.savefig(output_path, dpi=200)
  plt.close(fig)


@click.command()
@click.option("--model_name", type=click.Choice(MODELS), required=True)
@click.option("--dataset_name", type=click.Choice(DATASETS), required=True)
@click.option("--results_csv", type=str, default=DEFAULT_RESULTS_CSV, show_default=True)
@click.option("--payoffs", type=str, default=",".join(ALL_PAYOFFS), show_default=True)
@click.option("--coverage_threshold", type=float, default=0.5, show_default=True)
def main(model_name, dataset_name, results_csv, payoffs, coverage_threshold):
  payoffs = payoffs.split(",")
  output_dir = os.path.join(RESULTS_DIR, "e2", f"{dataset_name}_{model_name}_{MASKER}")
  os.makedirs(output_dir, exist_ok=True)

  wide = load_wide(results_csv, dataset_name, model_name, MASKER, payoffs)

  for metric in ["spearman", "sign_disagreement"]:
    mean_matrix, coverage_matrix = build_matrices(wide, payoffs, metric)

    out_path = os.path.join(output_dir, f"e2_{metric}_matrix.pdf")
    plot_matrix(mean_matrix, payoffs, metric, out_path, coverage_matrix, coverage_threshold)

    cov_csv_path = os.path.join(output_dir, f"e2_{metric}_coverage.csv")
    pd.DataFrame(coverage_matrix, index=payoffs, columns=payoffs).to_csv(cov_csv_path)


if __name__ == "__main__":
  main()