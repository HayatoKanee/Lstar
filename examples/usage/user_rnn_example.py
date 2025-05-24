#!/usr/bin/env python3
"""
User RNN Example: Using Any RNN with L* Algorithm

This script demonstrates how to use the L* algorithm with any user-provided RNN.
It shows examples for different types of RNNs and how to adapt them to work
with the L* framework.
"""

import time
import torch
import torch.nn as nn
from typing import List

from rnn_adapters import (
    PyTorchRNNAdapter, 
    CustomFunctionRNNAdapter, 
    RNNLoader,
    DummyRNN
)
from lstar.oracles import AnalyzingGenericRNNMembershipOracle
from lstar import WMethodEquivalenceOracle
from lstar import LStarLearner


def example_1_custom_function():
    """Example 1: Using a custom function as an RNN."""
    print("=== Example 1: Custom Function RNN ===")
    print("Learning DFA for strings that start and end with the same character\n")
    
    def same_start_end_classifier(string: str) -> float:
        """Returns high confidence if string starts and ends with same character."""
        if len(string) == 0:
            return 0.9  # Empty string trivially satisfies condition
        elif len(string) == 1:
            return 0.9  # Single character satisfies condition
        else:
            same = string[0] == string[-1]
            return 0.85 if same else 0.15
    
    alphabet = ["a", "b"]
    
    # Create RNN adapter
    rnn = CustomFunctionRNNAdapter(same_start_end_classifier, alphabet)
    
    # Test the function
    test_strings = ["", "a", "aa", "ab", "ba", "aba", "bab", "abc"]
    print("Testing custom function:")
    for s in test_strings:
        result = rnn.query(s)
        confidence = rnn.get_confidence(s)
        print(f"  '{s}': {result} (confidence: {confidence:.3f})")
    
    # Learn DFA using L*
    print("\nLearning DFA with L* algorithm...")
    membership_oracle = AnalyzingGenericRNNMembershipOracle(rnn)
    equivalence_oracle = WMethodEquivalenceOracle(membership_oracle, alphabet)
    learner = LStarLearner(alphabet, membership_oracle, equivalence_oracle)
    
    start_time = time.time()
    learned_dfa = learner.run()
    end_time = time.time()
    
    print(f"Learned DFA: {learned_dfa}")
    print(f"Learning time: {end_time - start_time:.3f} seconds")
    membership_oracle.print_query_statistics()
    
    return learned_dfa


def example_2_pytorch_model():
    """Example 2: Using a PyTorch model."""
    print("\n=== Example 2: PyTorch RNN Model ===")
    print("Training a simple RNN to recognize balanced parentheses\n")
    
    # Define a simple RNN model
    class BalancedParenthesesRNN(nn.Module):
        def __init__(self, vocab_size, hidden_size=16):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.classifier = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            embedded = self.embedding(x)
            output, hidden = self.rnn(embedded)
            # Use final hidden state for classification
            logits = self.classifier(hidden.squeeze(0))
            return torch.sigmoid(logits)
    
    alphabet = ["(", ")"]
    vocab_size = len(alphabet) + 1  # +1 for padding
    
    # Create and "train" the model (we'll use a simple rule-based approach for demo)
    model = BalancedParenthesesRNN(vocab_size)
    
    # Create custom tokenizer
    def paren_tokenizer(string: str) -> List[int]:
        char_to_idx = {"(": 1, ")": 2}
        return [char_to_idx.get(char, 0) for char in string]
    
    # Create RNN adapter
    rnn = PyTorchRNNAdapter(
        model, 
        alphabet, 
        tokenizer=paren_tokenizer,
        threshold=0.5
    )
    
    # Test the model (note: untrained model will give random results)
    test_strings = ["", "()", "((", "))", "(())", "()()"]
    print("Testing PyTorch model (untrained - random results expected):")
    for s in test_strings:
        result = rnn.query(s)
        confidence = rnn.get_confidence(s)
        print(f"  '{s}': {result} (confidence: {confidence:.3f})")
    
    # Learn DFA using L*
    print("\nLearning DFA with L* algorithm...")
    membership_oracle = AnalyzingGenericRNNMembershipOracle(rnn)
    equivalence_oracle = WMethodEquivalenceOracle(membership_oracle, alphabet)
    learner = LStarLearner(alphabet, membership_oracle, equivalence_oracle)
    
    start_time = time.time()
    learned_dfa = learner.run()
    end_time = time.time()
    
    print(f"Learned DFA: {learned_dfa}")
    print(f"Learning time: {end_time - start_time:.3f} seconds")
    membership_oracle.print_query_statistics()
    
    return learned_dfa


def example_3_dummy_rnn_adapter():
    """Example 3: Using our DummyRNN through the generic interface."""
    print("\n=== Example 3: DummyRNN via Generic Interface ===")
    print("Using DummyRNN through the generic interface\n")
    
    alphabet = ["0", "1"]
    pattern = r'^(0|1)*10(0|1)*$'  # Contains substring "10"
    
    # Create and train DummyRNN
    dummy_rnn = DummyRNN(alphabet, pattern)
    
    # Adapt to generic interface
    rnn = RNNLoader.from_dummy_rnn(dummy_rnn)
    
    # Test the adapter
    test_strings = ["", "0", "1", "10", "01", "110", "101", "010"]
    print("Testing DummyRNN via generic interface:")
    for s in test_strings:
        result = rnn.query(s)
        confidence = rnn.get_confidence(s)
        print(f"  '{s}': {result} (confidence: {confidence:.3f})")
    
    # Learn DFA using L*
    print("\nLearning DFA with L* algorithm...")
    membership_oracle = AnalyzingGenericRNNMembershipOracle(rnn)
    equivalence_oracle = WMethodEquivalenceOracle(membership_oracle, alphabet)
    learner = LStarLearner(alphabet, membership_oracle, equivalence_oracle)
    
    start_time = time.time()
    learned_dfa = learner.run()
    end_time = time.time()
    
    print(f"Learned DFA: {learned_dfa}")
    print(f"Learning time: {end_time - start_time:.3f} seconds")
    membership_oracle.print_query_statistics()
    membership_oracle.analyze_patterns()
    
    return learned_dfa


def example_4_loading_saved_model():
    """Example 4: How to load a saved PyTorch model."""
    print("\n=== Example 4: Loading Saved Model ===")
    print("Demonstrating how to load a saved PyTorch model\n")
    
    # This example shows the pattern, but doesn't actually load a file
    # since we don't have a pre-saved model
    
    print("Example code for loading a saved model:")
    print("""
    # Define your model class
    class MyRNNModel(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super().__init__()
            # ... model definition ...
        
        def forward(self, x):
            # ... forward pass ...
            return output
    
    # Load the model
    alphabet = ["a", "b", "c"]
    rnn = RNNLoader.from_pytorch_file(
        model_path="path/to/your/model.pth",
        model_class=MyRNNModel,
        alphabet=alphabet,
        model_kwargs={"vocab_size": 10, "hidden_size": 32},
        threshold=0.5,
        device="cpu"
    )
    
    # Use with L* algorithm
    membership_oracle = AnalyzingGenericRNNMembershipOracle(rnn)
    equivalence_oracle = WMethodEquivalenceOracle(membership_oracle, alphabet)
    learner = LStarLearner(alphabet, membership_oracle, equivalence_oracle)
    learned_dfa = learner.run()
    """)


def compare_different_rnns():
    """Compare DFAs learned from different types of RNNs."""
    print("\n=== Comparison: Different RNN Types ===")
    
    alphabet = ["a", "b"]
    
    # Create different RNNs for the same concept (palindromes)
    def palindrome_fn(s: str) -> float:
        if len(s) <= 1:
            return 0.9
        return 0.8 if s == s[::-1] else 0.2
    
    # RNN 1: Custom function
    rnn1 = CustomFunctionRNNAdapter(palindrome_fn, alphabet)
    
    # RNN 2: DummyRNN trained on palindrome pattern
    # Note: This is a simplified example - actual palindrome regex is complex
    dummy_rnn = DummyRNN(alphabet, r'^(a|b|aa|bb|aba|bab)$')  # Simple palindromes
    rnn2 = RNNLoader.from_dummy_rnn(dummy_rnn)
    
    rnns = [
        ("Custom Function", rnn1),
        ("DummyRNN", rnn2)
    ]
    
    dfas = []
    
    for name, rnn in rnns:
        print(f"\nLearning DFA from {name}...")
        
        membership_oracle = AnalyzingGenericRNNMembershipOracle(rnn)
        equivalence_oracle = WMethodEquivalenceOracle(membership_oracle, alphabet)
        learner = LStarLearner(alphabet, membership_oracle, equivalence_oracle)
        
        start_time = time.time()
        dfa = learner.run()
        end_time = time.time()
        
        dfas.append((name, dfa))
        
        print(f"  Learned DFA: {dfa}")
        print(f"  States: {len(dfa.states)}")
        print(f"  Learning time: {end_time - start_time:.3f} seconds")
        print(f"  Queries: {membership_oracle.get_query_count()}")
    
    # Compare the learned DFAs
    print(f"\n=== Comparison Results ===")
    for i, (name1, dfa1) in enumerate(dfas):
        for j, (name2, dfa2) in enumerate(dfas[i+1:], i+1):
            print(f"\nComparing {name1} vs {name2}:")
            
            # Test on sample strings
            test_strings = ["", "a", "b", "aa", "ab", "ba", "bb", "aba", "bab"]
            matches = 0
            
            for s in test_strings:
                result1 = dfa1.accepts(s)
                result2 = dfa2.accepts(s)
                if result1 == result2:
                    matches += 1
                else:
                    print(f"  Disagreement on '{s}': {name1}={result1}, {name2}={result2}")
            
            agreement = matches / len(test_strings) * 100
            print(f"  Agreement: {matches}/{len(test_strings)} ({agreement:.1f}%)")


def main():
    """Run all examples."""
    print("=== User RNN Integration with L* Algorithm ===")
    print("This example shows how to use ANY RNN with the L* framework\n")
    
    # Run examples
    example_1_custom_function()
    example_2_pytorch_model()
    example_3_dummy_rnn_adapter()
    example_4_loading_saved_model()
    compare_different_rnns()
    
    print("\n=== Summary ===")
    print("The generic RNN interface allows you to:")
    print("1. Use custom functions as 'RNNs'")
    print("2. Adapt PyTorch models with custom tokenizers")
    print("3. Load saved models from files")
    print("4. Work with Hugging Face transformers")
    print("5. Compare different RNN approaches")
    print("\nAll RNN types work seamlessly with the same L* algorithm!")


if __name__ == "__main__":
    main() 