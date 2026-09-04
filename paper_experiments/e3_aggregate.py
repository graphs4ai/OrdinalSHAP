import glob
import os
import numpy as np
import pandas as pd

RESULTS_DIR = "results"
E3_DIR = os.path.join(RESULTS_DIR, "e3")

ALPHA = 0.05

CHANCE_METHODS = {"random", "constant-positive", "constant-negative"}


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


def correct_family(df, mask, label):
  sub = df.loc[mask]
  valid = sub["p_value"].notna()
  n_valid = int(valid.sum())
  n_missing = int((~valid).sum())

  print()
  print("=" * 80)
  print(f"FAMILY: {label}")
  print("=" * 80)
  print(f"  tests: {len(sub)}  ({n_valid} with a defined p-value)")

  if n_missing:
    print(f"  {n_missing} tests with an undefined p-value, excluded from the correction.")

  if n_valid == 0:
    return

  idx = sub.index[valid]
  df.loc[idx, "p_value_holm_global"] = holm_bonferroni(df.loc[idx, "p_value"].values)
  df.loc[mask, "family"] = label
  df.loc[mask, "n_tests_in_family"] = n_valid

  holm_floor = ALPHA / n_valid
  df.loc[mask, "holm_floor"] = holm_floor
  df.loc[mask, "underpowered"] = df.loc[mask, "min_attainable_p"] > holm_floor

  n_under = int(df.loc[mask, "underpowered"].fillna(False).sum())
  print(f"  Holm's most permissive threshold: {holm_floor:.3e}")
  print(f"  requires at least {int(np.ceil(np.log2(2 / holm_floor)))} non-zero paired differences")
  print(f"  tests that cannot reach significance by design: {n_under}/{n_valid}")


def summarize(df, mask, label):
  sub = df.loc[mask].copy()
  if sub.empty:
    return

  sub["testable"] = sub["p_value"].notna() & ~sub["underpowered"].fillna(False)

  summary = sub.groupby(["target", "metric", "comparison"]).agg(
      n_significant=("significant_holm_global", "sum"),
      n_testable=("testable", "sum"),
      n_configs=("comparison", "size"),
      n_underpowered=("underpowered", lambda s: int(s.fillna(False).sum())),
      n_undefined=("p_value", lambda s: int(s.isna().sum())),
      median_effect=("effect_size", "median"),
    ).reset_index()

  print()
  print("=" * 80)
  print(f"SIGNIFICANCE COUNTS — {label}")
  print("=" * 80)
  print(summary.to_string(index=False))


def main():
  df = load_all_raw_wilcoxon()

  n_configs = df[["model", "dataset", "masker"]].drop_duplicates().shape[0]
  print(f"Pooled {len(df)} tests across {n_configs} configurations.")

  for col in ["p_value_holm_global", "holm_floor", "n_tests_in_family"]:
    df[col] = np.nan
  df["underpowered"] = pd.NA
  df["family"] = pd.NA

  is_chance = df["method"].isin(CHANCE_METHODS)

  correct_family(df, ~is_chance, "payoff comparisons")
  correct_family(df, is_chance, "chance baselines (diagnostic)")

  df["significant_holm_global"] = df["p_value_holm_global"] < ALPHA

  out_path = os.path.join(E3_DIR, "e3_wilcoxon_global.csv")
  df.to_csv(out_path, index=False)

  summarize(df, ~is_chance, "payoff comparisons")
  summarize(df, is_chance, "chance baselines (diagnostic)")

  under = df[df["underpowered"].fillna(False)]
  if len(under):
    print()
    print("=" * 80)
    print("TESTS THAT CANNOT REACH SIGNIFICANCE BY DESIGN")
    print("=" * 80)
    cols = ["family", "dataset", "model", "target", "metric", "method", "n", "n_nonzero", "min_attainable_p", "p_value", "effect_size"]
    print(under[cols].to_string(index=False))

  undefined = df[df["p_value"].isna()]
  if len(undefined):
    print()
    print("=" * 80)
    print("TESTS WITH NO NON-ZERO PAIRED DIFFERENCES")
    print("=" * 80)
    cols = ["family", "dataset", "model", "target", "metric", "method", "n"]
    print(undefined[cols].to_string(index=False))


def pool_diagnostics():
  paths = sorted(glob.glob(os.path.join(E3_DIR, "*", "e3_diagnostics.csv")))
  if not paths:
    print(f"No e3_diagnostics.csv files found under {E3_DIR}. Skipping.")
    return

  frames = [pd.read_csv(p) for p in paths]
  df = pd.concat(frames, ignore_index=True)
  out_path = os.path.join(E3_DIR, "e3_diagnostics_global.csv")
  df.to_csv(out_path, index=False)
  print(f"Diagnostics pooled from {len(paths)} configurations -> {out_path}")

  fallback = df[df["diagnostic"] == "ordinal_value_fallback_count"]["value"].sum()
  total = df[df["diagnostic"] == "ordinal_value_total_calls"]["value"].sum()
  if total:
    print(f"  Ordinal-value fallback across all configurations: "
          f"{int(fallback)}/{int(total)} = {fallback / total:.5%}")

  eff = df[df["diagnostic"] == "n_efficiency_failures"]["value"].sum()
  print(f"  Shapley efficiency violations: {int(eff)}")


def pool_summaries():
  paths = sorted(glob.glob(os.path.join(E3_DIR, "*", "e3_fidelity_summary.csv")))
  if not paths:
    return

  frames = []
  for path in paths:
    part = pd.read_csv(path)
    cfg = os.path.basename(os.path.dirname(path))
    part["configuration"] = cfg
    frames.append(part)
  df = pd.concat(frames, ignore_index=True)

  out_path = os.path.join(E3_DIR, "e3_fidelity_summary_global.csv")
  df.to_csv(out_path, index=False)

  pooled = df.groupby(["method", "metric"]).agg(
      mean_of_means=("mean", "mean"),
      std_of_means=("mean", "std"),
      mean_coverage=("coverage", "mean"),
      min_coverage=("coverage", "min"),
      n_configs=("mean", "size"),
    ).reset_index()
  pooled.to_csv(os.path.join(E3_DIR, "e3_fidelity_pooled.csv"), index=False)

  print()
  print("=" * 80)
  print("FIDELITY POOLED ACROSS CONFIGURATIONS")
  print("=" * 80)
  for metric in ["sa_dir", "rc_dir", "mae_dir", "sa_pres", "rc_pres", "mae_pres"]:
    sub = pooled[pooled["metric"] == metric].sort_values("mean_of_means", ascending=False)
    print(f"\n{metric}")
    print(sub[["method", "mean_of_means", "std_of_means", "mean_coverage", "min_coverage"]].to_string(index=False))


if __name__ == "__main__":
  main()
  pool_diagnostics()
  pool_summaries()
