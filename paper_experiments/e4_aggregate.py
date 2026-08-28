import glob
import os

import numpy as np
import pandas as pd

RESULTS_DIR = "results"
E4_DIR = os.path.join(RESULTS_DIR, "e4")


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
  paths = sorted(glob.glob(os.path.join(E4_DIR, "*", "e4_wilcoxon_raw.csv")))
  if not paths:
    raise FileNotFoundError(f"No e4_wilcoxon_raw.csv files found under {E4_DIR}. Run e4.py for every (model, dataset) configuration first.")
  frames = [pd.read_csv(p) for p in paths]
  return pd.concat(frames, ignore_index=True)


def main():
  df = load_all_raw_wilcoxon()

  n_configs = df[["model", "dataset"]].drop_duplicates().shape[0]
  n_tests = len(df)
  print(f"Pooled {n_tests} tests across {n_configs} configurations.")

  p_filled = df["p_value"].fillna(1.0).values
  df["p_value_holm_global"] = holm_bonferroni(p_filled)
  df["significant_holm_global"] = df["p_value_holm_global"] < 0.05

  out_path = os.path.join(E4_DIR, "e4_wilcoxon_global.csv")
  df.to_csv(out_path, index=False)

  print()
  print("=" * 80)
  print("PER-METRIC SIGNIFICANCE COUNT")
  print("=" * 80)
  summary = df.groupby("metric")["significant_holm_global"].agg(["sum", "count"]).rename(columns={"sum": "n_significant", "count": "n_configs_tested"})
  print(summary.to_string())


def load_all_agreement_and_fidelity():
  agreement_paths = sorted(glob.glob(os.path.join(E4_DIR, "*", "e4_masker_agreement_summary.csv")))
  fidelity_paths = sorted(glob.glob(os.path.join(E4_DIR, "*", "e4_fidelity_summary.csv")))

  if agreement_paths:
    frames = []
    for p in agreement_paths:
      config = os.path.basename(os.path.dirname(p))
      df = pd.read_csv(p)
      df["config"] = config
      frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(E4_DIR, "e4_masker_agreement_global.csv")
    combined.to_csv(out_path, index=False)
    print(f"Masker agreement pooled from {len(agreement_paths)} configurations -> {out_path}")
  else:
    print("No e4_masker_agreement_summary.csv files found -- skipping.")

  if fidelity_paths:
    frames = []
    for p in fidelity_paths:
      config = os.path.basename(os.path.dirname(p))
      df = pd.read_csv(p)
      df["config"] = config
      frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(E4_DIR, "e4_fidelity_global.csv")
    combined.to_csv(out_path, index=False)
    print(f"Fidelity pooled from {len(fidelity_paths)} configurations -> {out_path}")
  else:
    print("No e4_fidelity_summary.csv files found -- skipping.")


if __name__ == "__main__":
  main()
  load_all_agreement_and_fidelity()