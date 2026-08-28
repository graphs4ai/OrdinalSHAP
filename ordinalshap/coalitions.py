import random

class PermutationSampler:

  def __init__(self, n_samples: int = 1000, random_state: int = 42):
    if n_samples < 0:
      raise ValueError("n_samples must be non-negative.")

    self.n_samples = n_samples
    self.random = random.Random(random_state)

  def sample(self, n_features: int):

    if n_features < 0:
      raise ValueError("n_features must be non-negative.")

    features = list(range(n_features))
    for _ in range(self.n_samples):
      permutation = features.copy()
      self.random.shuffle(permutation)
      yield tuple(permutation)