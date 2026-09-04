import glob
import os

import numpy as np
import pandas as pd

RESULTS_DIR = "results"
E4_DIR = os.path.join(RESULTS_DIR, "e4")

ALPHA = 0.05


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
    raise FileNotFoundError(f"No e4_wilcoxon_raw.csv files found under {E4_DIR}. Run e4.py for every (model, dataset, payoff) configuration first.")
  frames = [pd.read_csv(p) for p in paths]
  return pd.concat(frames, ignore_index=True)


def apply_holm(df, mask, column, label):
  """Apply Holm within a subset and report what could not be tested."""
  sub = df.loc[mask]
  if sub.empty:
    return

  valid = sub["p_value"].notna()
  n_valid = int(valid.sum())
  n_missing = int((~valid).sum())

  print()
  print(f"  {label}: {len(sub)} tests ({n_valid} with a defined p-value)")
  if n_missing:
    print(f"    {n_missing} excluded from the correction, with no defined p-value")
  if n_valid == 0:
    return

  idx = sub.index[valid]
  df.loc[idx, column] = holm_bonferroni(df.loc[idx, "p_value"].values)

  floor = ALPHA / n_valid
  under = df.loc[mask, "min_attainable_p"] > floor
  n_under = int(under.fillna(False).sum())
  n_sig = int((df.loc[mask, column] < ALPHA).sum())

  print(f"    Holm's most permissive threshold: {floor:.3e}")
  print(f"    requires at least {int(np.ceil(np.log2(2 / floor)))} non-zero paired differences")
  print(f"    cannot reach significance by design: {n_under}/{n_valid}")
  print(f"    significant at alpha={ALPHA}: {n_sig}/{n_valid}")


def main():
  df = load_all_raw_wilcoxon()

  n_configs = df[["model", "dataset", "payoff"]].drop_duplicates().shape[0]
  n_tests = len(df)
  print(f"Pooled {n_tests} tests across {n_configs} configurations.")

  df["p_value_holm_global"] = np.nan
  df["p_value_holm_per_payoff"] = np.nan

  print()
  print("=" * 80)
  print("SCOPE A: SINGLE FAMILY ACROSS BOTH PAYOFFS")
  print("=" * 80)
  apply_holm(df, df.index.notna(), "p_value_holm_global", "all tests")

  print()
  print("=" * 80)
  print("SCOPE B: ONE FAMILY PER PAYOFF")
  print("=" * 80)
  for payoff in sorted(df["payoff"].unique()):
    apply_holm(df, df["payoff"] == payoff, "p_value_holm_per_payoff", payoff)

  df["significant_holm_global"] = df["p_value_holm_global"] < ALPHA
  df["significant_holm_per_payoff"] = df["p_value_holm_per_payoff"] < ALPHA

  n_valid_global = int(df["p_value"].notna().sum())
  df["underpowered_global"] = df["min_attainable_p"] > ALPHA / n_valid_global
  df["underpowered_per_payoff"] = False
  for payoff in df["payoff"].unique():
    mask = df["payoff"] == payoff
    n_valid = int(df.loc[mask, "p_value"].notna().sum())
    df.loc[mask, "underpowered_per_payoff"] = df.loc[mask, "min_attainable_p"] > ALPHA / n_valid

  out_path = os.path.join(E4_DIR, "e4_wilcoxon_global.csv")
  df.to_csv(out_path, index=False)

  print()
  print("=" * 80)
  print("PER-PAYOFF/METRIC SIGNIFICANCE COUNT")
  print("  scope A pools the two payoffs; scope B corrects each separately")
  print("=" * 80)
  summary = (
    df.groupby(["payoff", "metric"])
    .agg(
      n_sig_scope_a=("significant_holm_global", "sum"),
      n_sig_scope_b=("significant_holm_per_payoff", "sum"),
      n_underpowered_a=("underpowered_global", "sum"),
      n_underpowered_b=("underpowered_per_payoff", "sum"),
      n_configs_tested=("metric", "size"),
      median_effect=("effect_size", "median"),
    )
    .reset_index()
  )
  print(summary.to_string(index=False))

  # Tests whose outcome depends on the correction scope. These are the ones
  # that have to be reported explicitly rather than summarized as a count.
  disputed = df[df["significant_holm_global"] != df["significant_holm_per_payoff"]]
  if len(disputed):
    print()
    print("=" * 80)
    print("TESTS WHOSE SIGNIFICANCE DEPENDS ON THE CORRECTION SCOPE")
    print("=" * 80)
    cols = ["dataset", "model", "payoff", "metric", "p_value",
            "p_value_holm_global", "p_value_holm_per_payoff", "effect_size", "n", "n_nonzero"]
    print(disputed[cols].to_string(index=False))

  under = df[df["underpowered_global"].fillna(False)]
  if len(under):
    print()
    print("=" * 80)
    print("TESTS THAT CANNOT REACH SIGNIFICANCE BY DESIGN (SCOPE A)")
    print("=" * 80)
    cols = ["dataset", "model", "payoff", "metric", "n", "n_nonzero",
            "min_attainable_p", "p_value", "effect_size"]
    print(under[cols].to_string(index=False))

  print()
  print(f"Full table written to {out_path}")


def load_all_agreement_and_fidelity():
  agreement_paths = sorted(glob.glob(os.path.join(E4_DIR, "*", "e4_masker_agreement_summary.csv")))
  fidelity_paths = sorted(glob.glob(os.path.join(E4_DIR, "*", "e4_fidelity_summary.csv")))

  if agreement_paths:
    frames = []
    for p in agreement_paths:
      config = os.path.basename(os.path.dirname(p))
      df = pd.read_csv(p)
      df["config"] = config
      df["payoff"] = config.split("_", 2)[-1]
      frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(E4_DIR, "e4_masker_agreement_global.csv")
    combined.to_csv(out_path, index=False)
    print(f"Masker agreement pooled from {len(agreement_paths)} configurations -> {out_path}")

    pooled = (
      combined.groupby(["payoff", "metric"])
      .agg(mean_of_means=("mean", "mean"), std_of_means=("mean", "std"),
           mean_coverage=("coverage", "mean"), min_coverage=("coverage", "min"),
           n_configs=("mean", "size"))
      .reset_index()
    )
    pooled.to_csv(os.path.join(E4_DIR, "e4_masker_agreement_pooled.csv"), index=False)
    print()
    print("=" * 80)
    print("MASKER AGREEMENT POOLED ACROSS CONFIGURATIONS")
    print("=" * 80)
    print(pooled.to_string(index=False))
  else:
    print("No e4_masker_agreement_summary.csv files found -- skipping.")

  if fidelity_paths:
    frames = []
    for p in fidelity_paths:
      config = os.path.basename(os.path.dirname(p))
      df = pd.read_csv(p)
      df["config"] = config
      df["payoff"] = config.split("_", 2)[-1]
      frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(E4_DIR, "e4_fidelity_global.csv")
    combined.to_csv(out_path, index=False)
    print(f"Fidelity pooled from {len(fidelity_paths)} configurations -> {out_path}")

    pooled = (
      combined.groupby(["payoff", "masker", "metric"])
      .agg(mean_of_means=("mean", "mean"), std_of_means=("mean", "std"),
           mean_coverage=("coverage", "mean"), min_coverage=("coverage", "min"),
           n_configs=("mean", "size"))
      .reset_index()
    )
    pooled.to_csv(os.path.join(E4_DIR, "e4_fidelity_pooled.csv"), index=False)
    print()
    print("=" * 80)
    print("FIDELITY POOLED ACROSS CONFIGURATIONS")
    print("  Each payoff is evaluated on the target it decomposes.")
    print("=" * 80)
    print(pooled.to_string(index=False))
  else:
    print("No e4_fidelity_summary.csv files found -- skipping.")


if __name__ == "__main__":
  main()
  load_all_agreement_and_fidelity()
