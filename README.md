# Order Matters: Payoff Choice and the Explanatory Question in Shapley Explanations of Ordinal Text Classification

This repository contains the code for the paper "Order Matters: Payoff Choice and the Explanatory Question in Shapley Explanations of Ordinal Text Classification".

## Abstract

Shapley value explanations for large language models typically define a token coalition's value by the similarity between the model's response to that coalition and to the full input, an approach used by TokenSHAP and llmSHAP. On classification tasks with an ordinal label space, such payoffs quantify preservation of the original response but do not encode the direction of movement along the label scale. We argue that the choice of payoff determines which explanatory question a Shapley explanation answers, and we make that choice explicit by defining two payoffs on the ordinal scale: a directional payoff, whose value is a coalition's position on the scale, and a preservation payoff, whose value is the rank distance to the full-input response. We compare both against seven similarity-based payoffs across three language models and three evaluation sets derived from two sentiment datasets. Similarity payoffs are not reliably monotonic in rank distance: they degenerate on digit-based labels, and a widely used sentence embedder inverts the ordering on a Likert scale. The two ordinal payoffs produce attributions that differ from each other as much as either differs from the similarity payoffs, and the disagreement between the directional payoff and the similarity payoffs is determined by where the model's prediction falls on the scale rather than by the payoffs themselves. When fidelity is evaluated against an ordinal target, only the payoff aligned with that target decomposes it, and the similarity payoffs are matched by a baseline that assigns the same sign to every token. We release an open-source implementation, OrdinalSHAP, with a visualization module.

---

## Which payoff should I use?

OrdinalSHAP provides two payoff functions. They answer different questions about the same prediction, and neither is a better version of the other.

| | Question it answers | A positive attribution means |
| --- | --- | --- |
| `OrdinalDirectionalPayoff` | Which tokens move the prediction up or down the scale? | the token pushes the prediction toward higher-ranked labels |
| `OrdinalPreservationPayoff` | Which tokens keep the prediction where it is? | the token helps keep the response close to the full-input prediction |

The directional payoff is the one to use when you want to know what drove the prediction toward a given label. The preservation payoff answers the same question as TF-IDF, BM25 and embedding-based payoffs, and is the drop-in replacement for them on ordinal tasks: it measures distance in rank steps rather than in text space, so it cannot assign a higher similarity to a more distant label pair.

The two are related. When the full-input prediction sits at either end of the scale, `phi_pres = ∓phi_dir / (K-1)` exactly, so every sign is opposite (at the bottom) or identical (at the top). Elsewhere the relation is not affine and the two carry different information.

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
  InstanceModel, WordSplitter, DeleteMasker,
  OrdinalDirectionalPayoff, OrdinalPreservationPayoff,
  CoalitionEvaluator, PermutationSampler, ShapleyExplainer, ShapleyVisualization,
)

MAPPING = {"negative": 0, "neutral": 1, "positive": 2}
TEXT = "This movie was absolutely terrible"
SYSTEM_PROMPT = "Classify sentiment as negative, neutral, or positive. Output one word only."


async def explain(payoff, model, sampler):
  evaluator = CoalitionEvaluator(
    model=model,
    splitter=WordSplitter(),
    masker=DeleteMasker(),
    payoff=payoff,
    system_prompt=SYSTEM_PROMPT,
    text=TEXT,
  )
  await evaluator.initialize()
  explainer = ShapleyExplainer(evaluator=evaluator)
  phi = await explainer.explain(sampler.sample(len(evaluator.tokens)))
  return evaluator.tokens, explainer.to_array(phi)


async def main():
  model = InstanceModel(
    model_name="deepseek-ai/DeepSeek-V4-Flash-0731",
    api_key=os.environ["DEEPINFRA_API_KEY"],
    base_url="https://api.deepinfra.com/v1/openai",
  )
  # The same permutations are used for both payoffs, so the comparison is
  # paired and does not include variation from coalition sampling.
  sampler = PermutationSampler(n_samples=10, seed=0)

  # Which tokens move the prediction along the scale?
  tokens, phi_dir = await explain(OrdinalDirectionalPayoff(MAPPING), model, sampler)

  # Which tokens keep the prediction where it is?
  _, phi_pres = await explain(OrdinalPreservationPayoff(MAPPING), model, sampler)

  for token, d, p in zip(tokens, phi_dir, phi_pres):
    print(f"{token:<12} directional {d:+.3f}   preservation {p:+.3f}")

  ShapleyVisualization(tokens, phi_dir, label="Directional").plot_bar_chart("bar_directional.png")
  ShapleyVisualization(tokens, phi_pres, label="Preservation").plot_bar_chart("bar_preservation.png")


asyncio.run(main())
```

For a negative prediction, expect the two columns to carry opposite signs: a token that pushes the prediction downwards is also a token that helps preserve a negative prediction. Both attributions are correct, and which one you want depends on the question you are asking.

See `examples/README.md` and `examples/example.py` for the complete runnable version.

---

## Reproducing the Paper's Results

All experiments (E1-E5) can be reproduced from `paper_experiments/`. See `paper_experiments/README.md` for details, including which script produces which table or figure.

```bash
cd paper_experiments
python3 run_all.py
```

Note: this reruns all LLM API calls from scratch and can be costly. Pre-computed Shapley attributions (`paper_experiments/results/results.csv`) are provided so that the downstream analyses (E1-E5, without `compute_shapley.py`) can be reproduced offline, without any API access.

BM25 similarities use the BM25L variant. The Okapi formulation assigns a negative inverse document frequency to terms occurring in every document of the fitting corpus, which inverts the intended ordering when the labels share a verbalization template; results obtained with `BM25Okapi` will therefore differ from those reported in the paper.
