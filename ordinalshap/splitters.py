import re

class WordSplitter:

  def split(self, text: str):
    return re.findall(r"\b[\w]+(?:['’][\w]+)*\b", text)

  def join(self, tokens: list[str]):
    return " ".join(tokens)