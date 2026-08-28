import json

class CoalitionEvaluator:

  def __init__(
    self,
    model,
    splitter,
    masker,
    payoff,
    system_prompt: str | None = None,
    text: str = "",
  ):

    self.model = model
    self.splitter = splitter
    self.masker = masker
    self.payoff = payoff
    self.system_prompt = system_prompt
    self.text = text

    self.tokens = self.splitter.split(text)

    self.response_cache = {}
    self.value_cache = {}

    self.baseline_response = None

    self.full_coalition = frozenset(range(len(self.tokens)))


  async def initialize(self):

    if self.baseline_response is None:
      self.baseline_response = await self.model.generate(
        system_prompt=self.system_prompt,
        user_prompt=self.text,
      )
      self.response_cache[self.full_coalition] = self.baseline_response


  def _build_prompt(self, coalition: frozenset[int]):
    return self.masker.apply(self.tokens, coalition)


  async def response(self, coalition: frozenset[int]):

    coalition = frozenset(coalition)
    if coalition in self.response_cache:
      return self.response_cache[coalition]

    masked_text = self._build_prompt(coalition)

    response = await self.model.generate(
      system_prompt=self.system_prompt,
      user_prompt=masked_text,
    )

    self.response_cache[coalition] = response

    return response

  async def batch_response(self, coalitions):

    coalitions = [frozenset(c) for c in coalitions]
    missing = [c for c in coalitions if c not in self.response_cache]

    if missing:
      prompts = [self._build_prompt(c) for c in missing]

      responses = await self.model.batch_generate(
        prompts=prompts,
        system_prompt=self.system_prompt,
      )

      for coalition, response in zip(missing, responses):
        self.response_cache[coalition] = response

    return [self.response_cache[c] for c in coalitions]

  def value_from_response(self, coalition: frozenset[int], response: str):

    coalition = frozenset(coalition)
    if coalition in self.value_cache:
      return self.value_cache[coalition]

    value = self.payoff.evaluate(
      baseline_response=self.baseline_response,
      coalition_response=response,
    )

    self.value_cache[coalition] = value

    return value

  async def value(self, coalition: frozenset[int]):

    coalition = frozenset(coalition)
    if coalition in self.value_cache:
      return self.value_cache[coalition]

    response = await self.response(coalition)
    value = self.value_from_response(coalition, response)
    return value

  def save_responses(self, path):
    with open(path, "w") as f:
      json.dump({",".join(map(str, sorted(c))): r for c, r in self.response_cache.items()}, f)

  def load_responses(self, path, n_tokens):
    with open(path) as f:
      data = json.load(f)
    for key, response in data.items():
      coalition = frozenset(int(x) for x in key.split(",") if x != "")
      self.response_cache[coalition] = response
    if self.full_coalition in self.response_cache:
      self.baseline_response = self.response_cache[self.full_coalition]