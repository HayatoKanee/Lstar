import re


class MembershipOracle:
    def __init__(self, pattern: str):
        self._re = re.compile(pattern)

    def query(self, word: str) -> bool:
        """Return True iff `word` is in the target language."""
        return bool(self._re.fullmatch(word))