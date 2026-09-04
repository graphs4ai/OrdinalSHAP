import json
import os
import click
import numpy as np
import pandas as pd

from ordinalshap.visualization import ShapleyVisualization

CACHE_DIR = "response_cache"
RESULTS_DIR = "results"

DIRECTIONAL = "ordinal-directional"
PRESERVATION = "ordinal-preservation"

EMPTY_COALITION_RANK = 2


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


def ratio_summary(phi_dir, phi_pres, tol=1e-9):
  phi_dir = np.asarray(phi_dir, dtype=float)
  phi_pres = np.asarray(phi_pres, dtype=float)
  mask = np.abs(phi_dir) > tol
  if mask.sum() == 0:
    return np.nan, 0
  ratios = phi_pres[mask] / phi_dir[mask]
  distinct = len(np.unique(np.round(ratios, 6)))
  return float(np.median(ratios)), distinct


def collect_candidates(results_csv, model, dataset, masker, extra_payoff, min_response_diversity, n_samples):
  df = pd.read_csv(results_csv)
  df = df.drop_duplicates(subset=["dataset", "model", "masker", "payoff", "sample_id", "feature"], keep="last")
  subset = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["masker"] == masker)]

  wanted = [DIRECTIONAL, PRESERVATION]
  if extra_payoff:
    wanted.append(extra_payoff)

  frames = {}
  for payoff in wanted:
    part = subset[subset["payoff"] == payoff]
    if part.empty:
      raise ValueError(f"Payoff {payoff!r} not found for {model}/{dataset}/{masker}. Available: {sorted(subset['payoff'].unique())}")
    frames[payoff] = part

  candidates = []
  n_skipped_degenerate = 0
  n_skipped_no_cache = 0
  n_skipped_mismatch = 0

  for sample_id, g_dir in frames[DIRECTIONAL].groupby("sample_id"):
    parts = {DIRECTIONAL: g_dir}
    ok = True
    for payoff in wanted[1:]:
      g = frames[payoff][frames[payoff]["sample_id"] == sample_id]
      if len(g) != len(g_dir) or len(g) == 0:
        ok = False
        break
      parts[payoff] = g
    if not ok:
      n_skipped_mismatch += 1
      continue

    diversity = response_diversity(dataset, model, masker, sample_id, n_samples=n_samples)
    if diversity == 0:
      n_skipped_no_cache += 1
      continue
    if diversity < min_response_diversity:
      n_skipped_degenerate += 1
      continue

    merged = parts[DIRECTIONAL][["feature", "token", "shapley"]].rename(columns={"shapley": DIRECTIONAL})
    for payoff in wanted[1:]:
      col = parts[payoff][["feature", "shapley"]].rename(columns={"shapley": payoff})
      merged = merged.merge(col, on="feature")
    merged = merged.sort_values("feature")

    phi_dir = merged[DIRECTIONAL].values
    phi_pres = merged[PRESERVATION].values

    phi_sum = float(np.sum(phi_dir))
    o0 = int(round(phi_sum)) + EMPTY_COALITION_RANK
    integer_sum = abs(phi_sum - round(phi_sum)) < 1e-6

    median_ratio, n_distinct_ratios = ratio_summary(phi_dir, phi_pres)

    candidates.append({
      "sample_id": sample_id,
      "o0": o0,
      "phi_dir_sum": phi_sum,
      "integer_sum": integer_sum,
      "n_tokens": len(merged),
      "response_diversity": diversity,
      "sd_dir_pres": sign_disagreement(phi_dir, phi_pres),
      "median_ratio": median_ratio,
      "n_distinct_ratios": n_distinct_ratios,
      "affine_regime": n_distinct_ratios == 1,
      "tokens": merged["token"].tolist(),
      "values": {p: merged[p].values for p in wanted},
    })

  return candidates, wanted


def pick(candidates, o0_target, prefer_long=True):
  pool = [c for c in candidates if c["o0"] == o0_target]
  if not pool:
    available = sorted({c["o0"] for c in candidates})
    raise ValueError(f"No sample with o_0 = {o0_target}. Available values: {available}")
  key = (lambda c: (-c["n_tokens"], c["sample_id"])) if prefer_long \
      else (lambda c: (c["n_tokens"], c["sample_id"]))
  return sorted(pool, key=key)[0]


def report(selected, payoffs):
  print()
  print("=" * 78)
  print(f"sample_id={selected['sample_id']}  o_0={selected['o0']}  "
        f"n_tokens={selected['n_tokens']}  "
        f"response_diversity={selected['response_diversity']}")
  print("=" * 78)
  print(" ".join(selected["tokens"]))
  print()

  if selected["affine_regime"]:
    print(f"  Affine regime: phi_pres = {selected['median_ratio']:+.4f} * phi_dir for every")
    print("  token, which holds exactly when o_0 is at either end of the scale.")
    print(f"  Sign disagreement between the two payoffs: {selected['sd_dir_pres']:.3f}")
  else:
    print(f"  Non-affine regime: the ratio phi_pres/phi_dir takes "
          f"{selected['n_distinct_ratios']} distinct values across tokens, so the two")
    print("  payoffs are not related by a single constant on this sample.")
    print(f"  Sign disagreement between the two payoffs: {selected['sd_dir_pres']:.3f}")

  if not selected["integer_sum"]:
    print(f"  [warn] sum(phi_dir) = {selected['phi_dir_sum']:.6f} is not an integer; "
          f"o_0 may be misrecovered")

  print()
  header = f"  {'token':<18}" + "".join(f"{p:>24}" for p in payoffs)
  print(header)
  for i, token in enumerate(selected["tokens"]):
    row = f"  {token:<18}"
    for p in payoffs:
      row += f"{selected['values'][p][i]:>+24.3f}"
    print(row)


LABELS = {
  DIRECTIONAL: "Ordinal Directional",
  PRESERVATION: "Ordinal Preservation",
}


@click.command()
@click.option("--results_csv", type=str, default="results/results.csv")
@click.option("--model", type=str, default="deepseek")
@click.option("--dataset", type=str, default="sst-5")
@click.option("--masker", type=str, default="delete")
@click.option("--o0_targets", type=str, default="0")
@click.option("--baseline_payoff", type=str, default="")
@click.option("--top_k", type=int, default=10)
@click.option("--min_response_diversity", type=int, default=3)
@click.option("--n_samples", type=int, default=10)
def main(results_csv, model, dataset, masker, o0_targets, baseline_payoff, top_k, min_response_diversity, n_samples):
  output_dir = os.path.join(RESULTS_DIR, "e5")
  os.makedirs(output_dir, exist_ok=True)

  candidates, payoffs = collect_candidates(
    results_csv, model, dataset, masker, baseline_payoff or None,
    min_response_diversity, n_samples,
  )

  distribution = pd.Series([c["o0"] for c in candidates]).value_counts().sort_index()
  print(f"\no_0 distribution among candidates: {dict(distribution)}")

  rows = []
  for target in [int(t) for t in o0_targets.split(",")]:
    selected = pick(candidates, target)
    report(selected, payoffs)

    for payoff in payoffs:
      vis = ShapleyVisualization(
        selected["tokens"], selected["values"][payoff],
        label=LABELS.get(payoff, payoff.upper()),
      )
      path = os.path.join(
        output_dir,
        f"case_study_o0{target}_sample{selected['sample_id']}_{payoff}.pdf",
      )
      vis.plot_bar_chart(path, top_k=top_k)
      print(f"  wrote {path}")

    rows.append({k: v for k, v in selected.items() if k not in ("tokens", "values")})
    rows[-1]["text"] = " ".join(selected["tokens"])

  pd.DataFrame(rows).to_csv(os.path.join(output_dir, "e5_selected_samples.csv"), index=False)


if __name__ == "__main__":
  main()
