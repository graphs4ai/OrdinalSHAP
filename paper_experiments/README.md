# Paper Experiments

Scripts to reproduce every experiment (E1-E5) and every table/figure in the paper and its supplementary material.

**Run everything from inside this directory** (paths are relative to here):

```bash
cd paper_experiments
python3 run_all.py
```

`run_all.py` reruns the entire pipeline, including all LLM API calls, in order. This is costly (money and time) and non-deterministic in wall-clock terms (though sampling is seeded). If you only want to reproduce the downstream analyses (E1, E2, E3, E4 tables/figures) without calling any API, skip the `compute_shapley.py` calls - the pre-computed attributions in `results/results.csv` are already provided.

## Script → output map

| Script | Produces |
|---|---|
| `e1.py` | Ordinality preservation of baseline payoffs |
| `compute_shapley.py` | Shapley attributions (all model × payoff × dataset combinations) |
| `e2_heatmap.py` | Pairwise Spearman / sign disagreement matrices |
| `e2_analysis.py` | Per-configuration Spearman, sign disagreement, coverage |
| `e2_aggregate.py` | Pooled Wilcoxon tests (Holm-Bonferroni, global) |
| `e3.py` | Per-configuration sign agreement / magnitude correlation vs. payoffs |
| `e3_aggregate.py` | Pooled Wilcoxon tests, ordinal vs. baselines |
| `e4.py` | Masker agreement and fidelity (delete vs. replacement) |
| `e4_aggregate.py` | Pooled Wilcoxon tests, delete vs. replacement |
| `e5.py` | Qualitative case study (bar charts, ordinal vs. one baseline) |

## Dependencies between scripts

- `e1.py` is self-contained (no dependency on `compute_shapley.py`).
- `e2_*.py`, `e3.py`, `e4.py` all read from `results/results.csv`, produced by `compute_shapley.py`.
- `e5.py` additionally depends on the response cache (`results/response_cache/`) produced as a side effect of `compute_shapley.py`, using the **same** `--n_samples` value.

## Requirements

- `DEEPINFRA_API_KEY` - required for `compute_shapley.py`, `e3.py`, `e4.py`, `e5.py`.
- `GOOGLE_API_KEY` - required only if reproducing the `google-embeddings-2` payoff (`compute_shapley.py`, `e1.py`).
- Unix-like OS (uses `fcntl` for file locking in `compute_shapley.py`).