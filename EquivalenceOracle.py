from abc import ABC, abstractmethod
from collections import deque
from itertools import product
from typing import List, Set

from DFA import DFA
from MembershipOracle import MembershipOracle

# --- Abstract Equivalence Oracle ---
class EquivalenceOracle(ABC):
    @abstractmethod
    def find_counterexample(self, dfa: DFA) -> str | None:
        """
        Return a counterexample string where `dfa.accepts(s) != membership_oracle.query(s)`,
        or None if no discrepancy is found.
        """
        pass

# --- BFS Equivalence Oracle ---
class BFSEquivalenceOracle(EquivalenceOracle):
    def __init__(self, membership_oracle: MembershipOracle, alphabet: List[str], max_length: int = 8):
        self.membership_oracle = membership_oracle
        self.alphabet = alphabet
        self.max_length = max_length

    def find_counterexample(self, dfa: DFA) -> str | None:
        queue = deque([""])
        while queue:
            s = queue.popleft()
            if dfa.accepts(s) != self.membership_oracle.query(s):
                return s
            if len(s) < self.max_length:
                for a in self.alphabet:
                    queue.append(s + a)
        return None

# --- W-Method Equivalence Oracle ---
class WMethodEquivalenceOracle(EquivalenceOracle):
    def __init__(
        self,
        membership_oracle: MembershipOracle,
        alphabet: List[str],
        max_prefix_len: int = 2,
        max_suffix_len: int = 3,
    ):
        """
        - membership_oracle: black-box target language
        - alphabet: list of symbols
        - max_prefix_len: how deep to build the transition cover C
        - max_suffix_len: how deep to search for distinguishing suffixes
        """
        self.mo = membership_oracle
        self.alphabet = alphabet
        self.max_prefix_len = max_prefix_len
        self.max_suffix_len = max_suffix_len

    def find_counterexample(self, dfa: DFA) -> str | None:
        # 1) Build C = all strings of length <= max_prefix_len
        C: List[str] = [""]
        for L in range(1, self.max_prefix_len + 1):
            C.extend("".join(p) for p in product(self.alphabet, repeat=L))

        # 2) Build W = {""} ∪ all suffixes up to max_suffix_len that distinguish some state-pair
        W: Set[str] = {""}
        suffixes = [""]
        for L in range(1, self.max_suffix_len + 1):
            suffixes.extend("".join(p) for p in product(self.alphabet, repeat=L))

        n = len(dfa.states)
        for i in range(n):
            for j in range(i+1, n):
                for s in suffixes:
                    if self._accept_from(dfa, i, s) != self._accept_from(dfa, j, s):
                        W.add(s)
                        break

        # 3) Test all c+w for c in C, w in W
        for c in C:
            for w in sorted(W, key=lambda x:(len(x),x)):
                test = c + w
                if dfa.accepts(test) != self.mo.query(test):
                    return test

        return None

    def _accept_from(self, dfa: DFA, start: int, suffix: str) -> bool:
        st = start
        for ch in suffix:
            st = dfa.transitions[st][ch]
        return st in dfa.accepting_states
