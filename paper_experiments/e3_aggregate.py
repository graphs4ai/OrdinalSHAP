import glob
import os

import numpy as np
import pandas as pd

RESULTS_DIR = "results"
E3_DIR = os.path.join(RESULTS_DIR, "e3")

def holm_bonferroni(p_values):
  p_values = np.asarray(p_values, dtype=float)
  n = len(p_values)
  order = np.argsort(p_values)
  adjusted = np.empty(n)

  running_max = 0.0
  for rank, idx in enumerate(order):
    val = p_values[idx] * (n - rank)
    running_max = max(running_max, val)
    adjusted[idx] = min(running_max, 1.0)

  return adjusted


def load_all_raw_wilcoxon():
  paths = sorted(glob.glob(os.path.join(E3_DIR, "*", "e3_wilcoxon_raw.csv")))
  if not paths:
    raise FileNotFoundError(f"No e3_wilcoxon_raw.csv files found under {E3_DIR}. Run e3.py for every (model, dataset) configuration first.")

  frames = [pd.read_csv(p) for p in paths]
  df = pd.concat(frames, ignore_index=True)
  return df


def main():
  df = load_all_raw_wilcoxon()

  n_configs = df[["model", "dataset", "masker"]].drop_duplicates().shape[0]
  n_tests = len(df)
  print(f"Pooled {n_tests} tests across {n_configs} configurations.")

  p_filled = df["p_value"].fillna(1.0).values
  df["p_value_holm_global"] = holm_bonferroni(p_filled)
  df["significant_holm_global"] = df["p_value_holm_global"] < 0.05

  out_path = os.path.join(E3_DIR, "e3_wilcoxon_global.csv")
  df.to_csv(out_path, index=False)

  print()
  print("=" * 80)
  print("PER-METRIC/COMPARISON SIGNIFICANCE COUNT")
  print("=" * 80)
  summary = df.groupby(["metric", "comparison"])["significant_holm_global"].agg(["sum", "count"]).rename(columns={"sum": "n_significant", "count": "n_configs_tested"})
  print(summary.to_string())
  print()
  print(f"Full table written to {out_path}")


def load_all_diagnostics():
  paths = sorted(glob.glob(os.path.join(E3_DIR, "*", "e3_diagnostics.csv")))
  if not paths:
    print(f"No e3_diagnostics.csv files found under {E3_DIR}. Skipping diagnostics pooling.")
    return
  frames = [pd.read_csv(p) for p in paths]
  df = pd.concat(frames, ignore_index=True)
  out_path = os.path.join(E3_DIR, "e3_diagnostics_global.csv")
  df.to_csv(out_path, index=False)
  print(f"Diagnostics pooled from {len(paths)} configurations -> {out_path}")


if __name__ == "__main__":
  main()
  load_all_diagnostics()