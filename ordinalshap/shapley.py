class ShapleyExplainer:

  def __init__(self, evaluator):
    self.evaluator = evaluator

  async def explain(self, permutations):

    permutations = list(permutations)

    coalitions = set()
    for permutation in permutations:
      coalition = frozenset()
      coalitions.add(coalition)

      for feature in permutation:
        coalition = frozenset(set(coalition) | {feature})
        coalitions.add(coalition)

    await self.evaluator.batch_response(coalitions)

    if hasattr(self.evaluator.payoff, "fit"):
      self.evaluator.payoff.fit([self.evaluator.response_cache[c] for c in coalitions])

    values = {}
    for coalition in coalitions:
      response = self.evaluator.response_cache[coalition]
      values[coalition] = self.evaluator.value_from_response(coalition, response)

    phi = {i: 0.0 for i in range(len(self.evaluator.tokens))}

    n_permutations = len(permutations)
    for permutation in permutations:
      coalition = frozenset()

      previous_value = values[coalition]
      for feature in permutation:
        new_coalition = frozenset(set(coalition) | {feature})
        new_value = values[new_coalition]
        marginal_contribution = new_value - previous_value
        phi[feature] += marginal_contribution / n_permutations
        coalition = new_coalition
        previous_value = new_value

    return phi

  def to_array(self, phi):
    return [phi[i] for i in range(len(self.evaluator.tokens))]