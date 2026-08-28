# Examples

Minimal, self-contained usage examples for the `ordinalshap` package. Unlike `paper_experiments/`, these are not meant to reproduce any specific paper result - they exist to show a new user how to use the library on their own text and model.

## `example.py`

Runs the full OrdinalSHAP pipeline on a single hardcoded sentence, using the `ordinal` payoff (the method proposed in the paper) and the `delete` masker, and saves two visualizations of the resulting token attributions.

### Requirements

```bash
export DEEPINFRA_API_KEY=<your_key>
```

Any model available on DeepInfra can be used; the example defaults to `deepseek-ai/DeepSeek-V4-Flash-0731`.

### Run

```bash
cd examples
python3 example.py
```

### What it does

1. Splits the input text into tokens (`WordSplitter`).
2. Queries the model once per coalition sampled via `PermutationSampler`, masking out excluded tokens with `DeleteMasker`.
3. Maps each response onto the task's ordinal scale (`OrdinalPayoff`) and computes per-token Shapley values (`ShapleyExplainer`).
4. Saves two plots:
   - `bar.png` - bar chart of the top-8 tokens by |Shapley value|.
   - `text.png` - the full sentence, with each token color- and opacity-coded by its attribution.

### Adapting it to your own use case

To classify your own text with your own labels, change:

- `TEXT` — the input sentence.
- `SYSTEM_PROMPT` — instructions describing the classification task and the exact output format (the payoff parses the model's raw response, so the prompt must constrain the model to output only one of the label strings).
- `ORDINAL_MAPPING` — a dict mapping each possible label string to its position on the ordinal scale. The mapping only needs to be internally consistent (any constant offset cancels out in the Shapley computation) but the relative order must reflect the real ordinal scale of the task.

Swapping `DeleteMasker()` for `ReplacementMasker(replacement="[REMOVED]")` uses the replacement masking strategy (Section 4 / E4 of the paper) instead of deletion.