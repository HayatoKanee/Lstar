from graphviz import Digraph
from typing import List, Dict, Set

class DFA:
    def __init__(
        self,
        states: List[str],
        alphabet: List[str],
        start_state: int,
        accepting_states: Set[int],
        transitions: Dict[int, Dict[str, int]],
    ):
        self.states = states
        self.alphabet = alphabet
        self.start_state = start_state
        self.accepting_states = accepting_states
        self.transitions = transitions

    def accepts(self, s: str) -> bool:
        """Run the DFA on s and return True if in an accepting state."""
        state = self.start_state
        for ch in s:
            state = self.transitions[state][ch]
        return state in self.accepting_states

    def __repr__(self):
        return (
            f"<DFA states={len(self.states)} "
            f"start={self.start_state} accept={self.accepting_states}>"
        )

    def to_graphviz(self) -> Digraph:
        """Convert the DFA into a Graphviz Digraph object (PNG format by default)."""
        dot = Digraph(format="png")
        dot.attr(rankdir="LR")
        dot.node("__start__", shape="point")
        for i, label in enumerate(self.states):
            shape = "doublecircle" if i in self.accepting_states else "circle"
            dot.node(str(i), label=label, shape=shape)
        dot.edge("__start__", str(self.start_state))
        for src, trans in self.transitions.items():
            for sym, tgt in trans.items():
                dot.edge(str(src), str(tgt), label=sym)
        return dot

    def write_png(self, filename: str = "dfa") -> None:
        dot = self.to_graphviz()
        dot.render(filename, cleanup=False)
