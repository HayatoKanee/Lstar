# --- Main ---
import subprocess
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lstar import *
from lstar.oracles import RegexMembershipOracle

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