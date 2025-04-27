from typing import List, Set, Dict

from DFA import DFA
from MembershipOracle import MembershipOracle


class ObservationTable:
    def __init__(self, alphabet: List[str], membership_oracle: MembershipOracle):
        self.alphabet = alphabet
        self.membership_oracle = membership_oracle
        self.Q: Set[str] = {""}
        self.T: Set[str] = {""}
        self.M: Dict[str, Dict[str, bool]] = {}

    def get_membership(self, q: str, t: str) -> bool:
        if q not in self.M:
            self.M[q] = {}
        if t not in self.M[q]:
            val = self.membership_oracle.query(q + t)
            self.M[q][t] = val
            print(f"membership: '{q+t}' -> {val}")
        return self.M[q][t]

    def row(self, q: str) -> tuple[bool, ...]:
        return tuple(self.get_membership(q, t) for t in sorted(self.T))

    def find_distinguishing_suffix(self, v: str, w: str) -> str | None:
        for t in sorted(self.T):
            if self.get_membership(v, t) != self.get_membership(w, t):
                return t
        return None

    def make_consistent(self):
        while True:
            changed = False
            for q1 in list(self.Q):
                for q2 in list(self.Q):
                    if q1 < q2 and self.row(q1) == self.row(q2):
                        for a in self.alphabet:
                            v, w = q1 + a, q2 + a
                            diff_t = self.find_distinguishing_suffix(v, w)
                            if diff_t is not None:
                                new_t = a + diff_t
                                if new_t not in self.T:
                                    print(f"make_consistent: adding suffix '{new_t}'")
                                    self.T.add(new_t)
                                    for q in self.Q:
                                        self.get_membership(q, new_t)
                                    changed = True
                                break
                        if changed:
                            break
                if changed:
                    break
            if not changed:
                return

    def make_closed(self):
        while True:
            changed = False
            for q in list(self.Q):
                for a in self.alphabet:
                    qa = q + a
                    if not any(self.row(qa) == self.row(q2) for q2 in self.Q):
                        print(f"make_closed: adding prefix '{qa}'")
                        self.Q.add(qa)
                        for t in self.T:
                            self.get_membership(qa, t)
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                return

    def build_hypothesis(self) -> DFA:
        # Map each distinct row to exactly one representative prefix
        row_to_rep: Dict[tuple, str] = {}
        for q in self.Q:
            r = self.row(q)
            row_to_rep.setdefault(r, q)
        reps = list(row_to_rep.values())
        rep_to_id = {rep: idx for idx, rep in enumerate(reps)}

        # Build transitions and accepting sets
        transitions: Dict[int, Dict[str, int]] = {}
        accepting: Set[int] = set()
        for rep in reps:
            sid = rep_to_id[rep]
            transitions[sid] = {}
            if self.get_membership(rep, ""):
                accepting.add(sid)
            for a in self.alphabet:
                # find which rep matches rep+a
                succ = rep + a
                for r, rep2 in row_to_rep.items():
                    if self.row(succ) == r:
                        transitions[sid][a] = rep_to_id[rep2]
                        break

        return DFA(
            states=reps,
            alphabet=self.alphabet,
            start_state=rep_to_id[""],
            accepting_states=accepting,
            transitions=transitions,
        )