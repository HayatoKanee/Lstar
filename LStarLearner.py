from typing import List

from DFA import DFA
from EquivalenceOracle import EquivalenceOracle
from MembershipOracle import MembershipOracle
from ObservationTable import ObservationTable


class LStarLearner:
    def __init__(
        self,
        alphabet: List[str],
        membership_oracle: MembershipOracle,
        equivalence_oracle: EquivalenceOracle,
    ):
        self.table = ObservationTable(alphabet, membership_oracle)
        self.eq_oracle = equivalence_oracle

    def run(self) -> DFA:
        while True:
            self.table.make_consistent()
            self.table.make_closed()

            dfa = self.table.build_hypothesis()
            print("Hypothesis", dfa)

            ce = self.eq_oracle.find_counterexample(dfa)
            print("Counterexample:", ce)
            if ce is None:
                return dfa

            # refine Q with prefixes
            for i in range(1, len(ce) + 1):
                p = ce[:i]
                if p not in self.table.Q:
                    print(f"Refine Q with '{p}'")
                    self.table.Q.add(p)
                    for t in self.table.T:
                        self.table.get_membership(p, t)
