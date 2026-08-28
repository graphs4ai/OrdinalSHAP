class DeleteMasker:

  def apply(self, tokens: list[str], coalition: set[int]):

    kept = [token for i, token in enumerate(tokens) if i in coalition]
    if len(kept) == 0:
      return "[NO TEXT]"
    return " ".join(kept)


class ReplacementMasker:

  def __init__(self, replacement: str):
    self.replacement = replacement

  def apply(self, tokens: list[str], coalition: set[int]):

    output = []
    for i, token in enumerate(tokens):
      if i in coalition:
        output.append(token)
      else:
        output.append(self.replacement)
    return " ".join(output)