# Order Matters: Rethinking Shapley Payoffs for Ordinal Text Classification

This repository contains the code for the paper "Order Matters: Rethinking Shapley Payoffs for Ordinal Text Classification".

## Abstract

Shapley value explanations for large language models typically measure a token coalition's value by the similarity between the model's response to that coalition and to the full input, an approach used by TokenSHAP and llmSHAP. For classification tasks with an ordinal label space, this discards information the task provides, treating every change in prediction as equally important regardless of how far it moves along the scale. We propose a payoff function defined directly on a task's ordinal scale, so a coalition's value reflects its position on that scale rather than its similarity to the original response. We evaluate this ordinal payoff against seven similarity-based baselines, including those used by TokenSHAP and llmSHAP, across three language models and three sentiment datasets, asking whether existing payoffs preserve ordinal structure, whether the ordinal payoff yields different explanations, whether differences reflect greater fidelity to model behavior, and whether this fidelity is robust to the masking strategy. Existing payoffs preserve ordinal structure only partially and sometimes invert it. The ordinal payoff yields explanations structurally distinct from all similarity-based payoffs, is more faithful than every baseline in both direction and magnitude, and remains largely robust to deleting versus replacing masked tokens. We release an open-source implementation, OrdinalSHAP, with a visualization module for inspecting explanations on ordinal tasks.

---

## Repository Structure

```
.
├── ordinalshap/          # Core library: coalition sampling, maskers, payoffs, Shapley computation, visualization
├── data/                 # Datasets and generation script
├── paper_experiments/    # Scripts to reproduce every experiment and table/figure in the paper
├── examples/             # Minimal usage example for new users
├── environment.yml       # Conda environment
└── pyproject.toml        # Package metadata for `ordinalshap`
```

See `paper_experiments/README.md` for a script-by-script map to every table and figure in the paper, `data/README.md` for dataset provenance and licensing notes, and `examples/README.md` for a minimal usage walkthrough.

---

## Installation

### 1. Clone the repository

```bash
git clone <repo_url>
cd <repo>
```

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate ordinalshap
```

This also installs the `ordinalshap` package in editable mode. If you skip conda, you can install it manually:

```bash
pip install -e .
```

Optional baseline payoffs (TF-IDF, BM25, embedding-based) require extra dependencies:

```bash
pip install -e ".[baselines]"
```

### 3. API keys

Running the model or the Google embedding payoff requires API keys, exported as environment variables:

```bash
export DEEPINFRA_API_KEY=<your_key>
export GOOGLE_API_KEY=<your_key>   # only needed for the google-embeddings-2 payoff
```

See `.env.example` for the full list.

---

## Quickstart

```python
import asyncio
import os

from ordinalshap import (
  InstanceModel, WordSplitter, DeleteMasker, OrdinalDirectionalPayoff,
  CoalitionEvaluator, PermutationSampler, ShapleyExplainer, ShapleyVisualization,
)

async def main():
  model = InstanceModel(
    model_name="deepseek-ai/DeepSeek-V4-Flash-0731",
    api_key=os.environ["DEEPINFRA_API_KEY"],
    base_url="https://api.deepinfra.com/v1/openai",
  )
  evaluator = CoalitionEvaluator(
    model=model, 
    splitter=WordSplitter(), 
    masker=DeleteMasker(),
    payoff=OrdinalDirectionalPayoff({"negative": 0, "neutral": 1, "positive": 2}),
    system_prompt="Classify sentiment as negative, neutral, or positive. Output one word only.",
    text="This movie was absolutely terrible",
  )
  await evaluator.initialize()
  explainer = ShapleyExplainer(evaluator=evaluator)
  phi = await explainer.explain(PermutationSampler(n_samples=10).sample(len(evaluator.tokens)))

  vis = ShapleyVisualization(evaluator.tokens, explainer.to_array(phi))
  vis.plot_bar_chart("bar.png")

asyncio.run(main())
```

See `examples/README.md` and `examples/example.py` for the complete runnable version.

---

## Reproducing the Paper's Results

All experiments (E1-E5) can be reproduced from `paper_experiments/`. See `paper_experiments/README.md` for details, including which script produces which table or figure.

```bash
cd paper_experiments
python3 run_all.py
```

Note: this reruns all LLM API calls from scratch and can be costly. Pre-computed Shapley attributions (`paper_experiments/results/results.csv`) are provided so that the downstream analyses (E1-E5, without `compute_shapley.py`) can be reproduced offline, without any API access.
