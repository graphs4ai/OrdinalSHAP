import asyncio
import os

from ordinalshap import (
    InstanceModel,
    WordSplitter,
    DeleteMasker,
    OrdinalPayoff,
    CoalitionEvaluator,
    PermutationSampler,
    ShapleyExplainer,
    ShapleyVisualization,
)

TEXT = "This movie was absolutely terrible"

SYSTEM_PROMPT = """
You are performing a sentiment classification task. Your task is to classify the sentiment of the provided text. You MUST always select exactly one of the following categories:

very negative
negative
neutral
positive
very positive

Output ONLY one category: very negative, negative, neutral, positive, or very positive
"""

ORDINAL_MAPPING = {
  "very negative": 0,
  "negative": 1,
  "neutral": 2,
  "positive": 3,
  "very positive": 4,
}


async def main():
  model = InstanceModel(
    model_name="deepseek-ai/DeepSeek-V4-Flash-0731",
    api_key=os.environ["DEEPINFRA_API_KEY"],
    base_url="https://api.deepinfra.com/v1/openai",
    temperature=0.0,
    max_tokens=5,
    max_concurrency=8,
    requests_per_minute=50,
  )

  splitter = WordSplitter()
  masker = DeleteMasker()
  payoff = OrdinalPayoff(ORDINAL_MAPPING)
  sampler = PermutationSampler(n_samples=10)

  evaluator = CoalitionEvaluator(
    model=model,
    splitter=splitter,
    masker=masker,
    payoff=payoff,
    system_prompt=SYSTEM_PROMPT,
    text=TEXT,
  )
  await evaluator.initialize()

  permutations = sampler.sample(len(evaluator.tokens))
  explainer = ShapleyExplainer(evaluator=evaluator)
  phi = await explainer.explain(permutations=permutations)

  vis = ShapleyVisualization(
    tokens=evaluator.tokens,
    values=explainer.to_array(phi),
    label="Ordinal payoff",
  )

  vis.plot_bar_chart("bar.png", top_k=8)
  vis.plot_text("text.png")

if __name__ == "__main__":
  asyncio.run(main())