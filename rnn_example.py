#!/usr/bin/env python3
"""
Example: Using L* Algorithm to Learn DFA from RNN

This script demonstrates how to use the L* active learning algorithm
to extract a DFA from a trained RNN. The process involves:

1. Training a dummy RNN on a regular language pattern
2. Using the trained RNN as a membership oracle
3. Running L* with W-method equivalence checking
4. Analyzing the learned DFA and query statistics
"""

import time
from DummyRNN import DummyRNN
from RNNMembershipOracle import AnalyzingRNNMembershipOracle
from EquivalenceOracle import WMethodEquivalenceOracle
from LStarLearner import LStarLearner
import re


def compare_models(learned_dfa, original_pattern, alphabet, test_strings=None):
    """
    Compare the learned DFA with the original pattern on test strings.
    """
    if test_strings is None:
        # Generate comprehensive test strings
        test_strings = [""]  # empty string
        
        # All single characters
        test_strings.extend(alphabet)
        
        # All pairs
        for a in alphabet:
            for b in alphabet:
                test_strings.append(a + b)
        
        # Some longer strings
        longer_strings = [
            "000", "001", "010", "011", "100", "101", "110", "111",
            "0000", "0001", "0010", "0100", "1000", "0011", "0101", "1001",
            "01010", "10101", "00011", "11100", "010101", "101010"
        ]
        test_strings.extend(longer_strings)
    
    original_regex = re.compile(original_pattern)
    
    matches = 0
    mismatches = []
    
    print(f"\n=== Model Comparison ===")
    print(f"Testing {len(test_strings)} strings...")
    
    for s in test_strings:
        dfa_result = learned_dfa.accepts(s)
        regex_result = bool(original_regex.fullmatch(s))
        
        if dfa_result == regex_result:
            matches += 1
        else:
            mismatches.append((s, dfa_result, regex_result))
    
    accuracy = matches / len(test_strings) * 100
    print(f"Accuracy: {matches}/{len(test_strings)} ({accuracy:.1f}%)")
    
    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for s, dfa_res, regex_res in mismatches[:10]:  # Show first 10
            print(f"  '{s}': DFA={dfa_res}, Original={regex_res}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
    else:
        print("Perfect match! The learned DFA is equivalent to the original pattern.")
    
    return accuracy


def main():
    print("=== L* Algorithm: Learning DFA from RNN ===\n")
    
    # Configuration
    alphabet = ["0", "1"]
    pattern = r'^(0|1)*01(0|1)*$'  # Contains substring "01"
    
    print(f"Target pattern: {pattern}")
    print(f"Alphabet: {alphabet}")
    print(f"Language: Binary strings containing the substring '01'")
    
    # Step 1: Train the RNN
    print(f"\n--- Step 1: Training RNN ---")
    rnn = DummyRNN(alphabet, pattern)
    
    # Step 2: Create oracles
    print(f"\n--- Step 2: Setting up L* Framework ---")
    membership_oracle = AnalyzingRNNMembershipOracle(rnn)
    equivalence_oracle = WMethodEquivalenceOracle(
        membership_oracle, 
        alphabet,
        max_prefix_len=3,  # Increased for better coverage
        max_suffix_len=4   # Increased for better coverage
    )
    
    # Step 3: Run L* algorithm
    print(f"\n--- Step 3: Running L* Algorithm ---")
    learner = LStarLearner(alphabet, membership_oracle, equivalence_oracle)
    
    start_time = time.time()
    learned_dfa = learner.run()
    end_time = time.time()
    
    # Step 4: Analyze results
    print(f"\n--- Step 4: Analysis ---")
    print(f"Learning completed in {end_time - start_time:.2f} seconds")
    print(f"Learned DFA: {learned_dfa}")
    print(f"Number of states: {len(learned_dfa.states)}")
    print(f"Number of transitions: {sum(len(trans) for trans in learned_dfa.transitions.values())}")
    
    # Print query statistics
    membership_oracle.print_query_statistics()
    
    # Step 5: Validate the learned model
    print(f"\n--- Step 5: Validation ---")
    accuracy = compare_models(learned_dfa, pattern, alphabet)
    
    # Step 6: Test specific examples
    print(f"\n--- Step 6: Example Tests ---")
    test_cases = [
        ("", "Empty string"),
        ("0", "Single 0"),
        ("1", "Single 1"), 
        ("01", "Minimal pattern '01'"),
        ("10", "Reverse pattern '10'"),
        ("001", "Starts with 00, contains 01"),
        ("011", "Contains 01, ends with 1"),
        ("1001", "Contains both 10 and 01"),
        ("1010", "Alternating, contains 01"),
        ("0000", "All zeros"),
        ("1111", "All ones"),
        ("101010", "Long alternating"),
    ]
    
    print("String".ljust(12) + "DFA".ljust(8) + "RNN".ljust(8) + "Original".ljust(10) + "Description")
    print("-" * 60)
    
    original_regex = re.compile(pattern)
    for test_string, description in test_cases:
        dfa_result = learned_dfa.accepts(test_string)
        rnn_result = rnn.query(test_string)
        original_result = bool(original_regex.fullmatch(test_string))
        
        # Format results with checkmarks/crosses
        dfa_sym = "✓" if dfa_result else "✗"
        rnn_sym = "✓" if rnn_result else "✗"
        orig_sym = "✓" if original_result else "✗"
        
        print(f"'{test_string}'".ljust(12) + 
              f"{dfa_sym}".ljust(8) + 
              f"{rnn_sym}".ljust(8) + 
              f"{orig_sym}".ljust(10) + 
              description)
    
    # Step 7: Save the learned DFA
    print(f"\n--- Step 7: Saving Results ---")
    try:
        learned_dfa.write_png("learned_dfa_from_rnn")
        print("Saved DFA visualization to 'learned_dfa_from_rnn.png'")
    except Exception as e:
        print(f"Could not save visualization: {e}")
    
    print(f"\n=== Summary ===")
    print(f"Successfully learned a {len(learned_dfa.states)}-state DFA from RNN")
    print(f"Total membership queries: {membership_oracle.get_query_count()}")
    print(f"Model accuracy: {accuracy:.1f}%")
    print(f"Learning time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main() 