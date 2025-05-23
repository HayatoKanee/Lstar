# --- Main ---
from EquivalenceOracle import *
from LStarLearner import LStarLearner
from MembershipOracle import RegexMembershipOracle
import subprocess

if __name__ == "__main__":
    # Secret DFA: accepts binary strings containing the substring "01"
    regex = r'^(0|1)*01(0|1)*$'
    alphabet = ["0", "1"]

    # Create oracles.
    mo = RegexMembershipOracle(regex)
    eo = WMethodEquivalenceOracle(mo, alphabet)
    # eo = BFSEquivalenceOracle(mo, alphabet)
    learner = LStarLearner(alphabet, mo, eo)
    learned = learner.run()
    print("Learned DFA:", learned)
    dfa = learner.run()
    # dfa.write_dot("learned_dfa")
    dfa.write_png("learned_dfa")
    print("Wrote to learned_dfa.png") 