#!/usr/bin/env python3
"""
Comparison Example: L* Algorithm with Different Membership Oracles

This script demonstrates the flexibility of the L* framework by using it with
both RegexMembershipOracle and RNNMembershipOracle to learn the same language.
"""

import time
from DummyRNN import DummyRNN
from MembershipOracle import RegexMembershipOracle
from RNNMembershipOracle import AnalyzingRNNMembershipOracle
from EquivalenceOracle import WMethodEquivalenceOracle
from LStarLearner import LStarLearner


def learn_with_regex_oracle(pattern: str, alphabet: list):
    """Learn DFA using a regex-based membership oracle."""
    print("=== Learning with Regex Membership Oracle ===")
    
    # Create oracles
    membership_oracle = RegexMembershipOracle(pattern)
    equivalence_oracle = WMethodEquivalenceOracle(membership_oracle, alphabet)
    
    # Run L* algorithm
    learner = LStarLearner(alphabet, membership_oracle, equivalence_oracle)
    start_time = time.time()
    learned_dfa = learner.run()
    end_time = time.time()
    
    print(f"Learned DFA: {learned_dfa}")
    print(f"Learning time: {end_time - start_time:.3f} seconds")
    print(f"Number of states: {len(learned_dfa.states)}")
    
    return learned_dfa


def learn_with_rnn_oracle(pattern: str, alphabet: list):
    """Learn DFA using an RNN-based membership oracle."""
    print("\n=== Learning with RNN Membership Oracle ===")
    
    # Train RNN on the pattern
    print("Training RNN...")
    rnn = DummyRNN(alphabet, pattern)
    
    # Create oracles
    membership_oracle = AnalyzingRNNMembershipOracle(rnn)
    equivalence_oracle = WMethodEquivalenceOracle(membership_oracle, alphabet)
    
    # Run L* algorithm
    learner = LStarLearner(alphabet, membership_oracle, equivalence_oracle)
    start_time = time.time()
    learned_dfa = learner.run()
    end_time = time.time()
    
    print(f"Learned DFA: {learned_dfa}")
    print(f"Learning time: {end_time - start_time:.3f} seconds")
    print(f"Number of states: {len(learned_dfa.states)}")
    print(f"Total RNN queries: {membership_oracle.get_query_count()}")
    
    return learned_dfa, membership_oracle


def compare_dfas(dfa1, dfa2, alphabet: list, name1: str = "DFA1", name2: str = "DFA2"):
    """Compare two DFAs on a set of test strings."""
    print(f"\n=== Comparing {name1} vs {name2} ===")
    
    # Generate test strings
    test_strings = [""]  # empty string
    test_strings.extend(alphabet)  # single characters
    
    # All pairs
    for a in alphabet:
        for b in alphabet:
            test_strings.append(a + b)
    
    # Some longer strings
    longer_strings = [
        "000", "001", "010", "011", "100", "101", "110", "111",
        "0000", "0001", "0010", "0100", "1000", "01010", "10101"
    ]
    test_strings.extend(longer_strings)
    
    # Compare results
    matches = 0
    mismatches = []
    
    for s in test_strings:
        result1 = dfa1.accepts(s)
        result2 = dfa2.accepts(s)
        
        if result1 == result2:
            matches += 1
        else:
            mismatches.append((s, result1, result2))
    
    accuracy = matches / len(test_strings) * 100
    print(f"Agreement: {matches}/{len(test_strings)} ({accuracy:.1f}%)")
    
    if mismatches:
        print(f"Disagreements ({len(mismatches)}):")
        for s, res1, res2 in mismatches[:5]:  # Show first 5
            print(f"  '{s}': {name1}={res1}, {name2}={res2}")
        if len(mismatches) > 5:
            print(f"  ... and {len(mismatches) - 5} more")
    else:
        print("Perfect agreement! Both DFAs recognize the same language.")
    
    return accuracy


def main():
    print("=== L* Algorithm: Comparing Membership Oracle Types ===\n")
    
    # Configuration
    alphabet = ["0", "1"]
    pattern = r'^(0|1)*01(0|1)*$'  # Contains substring "01"
    
    print(f"Target pattern: {pattern}")
    print(f"Alphabet: {alphabet}")
    print(f"Language: Binary strings containing the substring '01'\n")
    
    # Learn with regex oracle
    regex_dfa = learn_with_regex_oracle(pattern, alphabet)
    
    # Learn with RNN oracle
    rnn_dfa, rnn_oracle = learn_with_rnn_oracle(pattern, alphabet)
    
    # Compare the learned DFAs
    accuracy = compare_dfas(regex_dfa, rnn_dfa, alphabet, "Regex-DFA", "RNN-DFA")
    
    # Test both DFAs on specific examples
    print(f"\n=== Example Tests ===")
    test_cases = [
        ("", "Empty string"),
        ("01", "Minimal pattern"),
        ("10", "Reverse pattern"),
        ("001", "Contains 01"),
        ("1001", "Contains both 10 and 01"),
        ("0000", "All zeros"),
        ("1111", "All ones"),
    ]
    
    print("String".ljust(10) + "Regex-DFA".ljust(12) + "RNN-DFA".ljust(12) + "Description")
    print("-" * 50)
    
    for test_string, description in test_cases:
        regex_result = regex_dfa.accepts(test_string)
        rnn_result = rnn_dfa.accepts(test_string)
        
        regex_sym = "✓" if regex_result else "✗"
        rnn_sym = "✓" if rnn_result else "✗"
        
        print(f"'{test_string}'".ljust(10) + 
              f"{regex_sym}".ljust(12) + 
              f"{rnn_sym}".ljust(12) + 
              description)
    
    # Save visualizations
    print(f"\n=== Saving Results ===")
    try:
        regex_dfa.write_png("learned_dfa_regex")
        print("Saved regex-learned DFA to 'learned_dfa_regex.png'")
        
        rnn_dfa.write_png("learned_dfa_rnn")
        print("Saved RNN-learned DFA to 'learned_dfa_rnn.png'")
    except Exception as e:
        print(f"Could not save visualizations: {e}")
    
    # Print RNN query statistics
    rnn_oracle.print_query_statistics()
    
    print(f"\n=== Summary ===")
    print(f"Both oracles successfully learned DFAs for the target language")
    print(f"Regex oracle: {len(regex_dfa.states)} states")
    print(f"RNN oracle: {len(rnn_dfa.states)} states")
    print(f"Agreement between learned DFAs: {accuracy:.1f}%")


if __name__ == "__main__":
    main() 