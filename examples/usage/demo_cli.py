#!/usr/bin/env python3
"""
Demonstration script showing how to use the lstar_extract.py command-line tool.

This script creates sample models and shows example commands for extracting DFAs.
"""

import os
import sys
import subprocess
import torch
import torch.nn as nn
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.append('.')

from examples.example_models import SimpleRNN
from DummyRNN import DummyRNN


def create_sample_pytorch_model():
    """Create and save a sample PyTorch model for demonstration."""
    print("Creating sample PyTorch model...")
    
    # Create a simple model
    vocab_size = 3  # For alphabet ["a", "b"]
    model = SimpleRNN(vocab_size, hidden_size=16)
    
    # Save the model
    model_path = "examples/sample_model.pth"
    os.makedirs("examples", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    print(f"Saved sample model to {model_path}")
    return model_path


def create_sample_trained_model():
    """Create and save a model trained on our DummyRNN for comparison."""
    print("Creating trained model using DummyRNN...")
    
    # Create DummyRNN
    alphabet = ["0", "1"]
    pattern = r'^(0|1)*10(0|1)*$'  # Contains substring "10"
    dummy_rnn = DummyRNN(alphabet, pattern)
    
    # Save the trained model
    trained_model_path = "examples/trained_dummy_model.pth"
    os.makedirs("examples", exist_ok=True)
    torch.save(dummy_rnn.model.state_dict(), trained_model_path)
    
    print(f"Saved trained model to {trained_model_path}")
    return trained_model_path, dummy_rnn


def run_command(cmd):
    """Run a command and print the output."""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"Exit code: {result.returncode}")
    except Exception as e:
        print(f"Error running command: {e}")


def demo_function_extraction():
    """Demonstrate DFA extraction from custom functions."""
    print("\n" + "="*80)
    print("DEMO 1: Extracting DFA from custom functions")
    print("="*80)
    
    # Example 1: Palindrome classifier
    cmd1 = """python lstar_extract.py --model-type function --model-path examples/simple_classifier.py --function-name palindrome_classifier --alphabet "abc" --output examples/palindrome_dfa.png --test-strings ",a,aa,ab,aba,abc,abcba" --verbose"""
    
    run_command(cmd1)
    
    # Example 2: Even length classifier
    cmd2 = """python lstar_extract.py --model-type function --model-path examples/simple_classifier.py --function-name even_length_classifier --alphabet "ab" --output examples/even_length_dfa.dot --test-strings ",a,aa,ab,aaa,aaaa" --verbose"""
    
    run_command(cmd2)
    
    # Example 3: Contains substring classifier
    cmd3 = """python lstar_extract.py --model-type function --model-path examples/simple_classifier.py --function-name contains_substring_classifier --alphabet "ab" --output examples/contains_ab_dfa.png --test-strings "a,ab,ba,aab,bab,abab" --verbose"""
    
    run_command(cmd3)


def demo_pytorch_extraction():
    """Demonstrate DFA extraction from PyTorch models."""
    print("\n" + "="*80)
    print("DEMO 2: Extracting DFA from PyTorch models")
    print("="*80)
    
    # Create sample model
    model_path = create_sample_pytorch_model()
    
    # Example 1: Simple untrained model
    cmd1 = f"""python lstar_extract.py --model-type pytorch --model-path {model_path} --model-class SimpleRNN --model-module examples/example_models.py --model-kwargs '{{"vocab_size": 3, "hidden_size": 16}}' --alphabet "ab" --output examples/pytorch_dfa.png --test-strings ",a,b,aa,ab,ba,bb" --verbose"""
    
    run_command(cmd1)
    
    # Example 2: With different threshold
    cmd2 = f"""python lstar_extract.py --model-type pytorch --model-path {model_path} --model-class SimpleRNN --model-module examples/example_models.py --model-kwargs '{{"vocab_size": 3, "hidden_size": 16}}' --alphabet "ab" --threshold 0.3 --output examples/pytorch_dfa_threshold.png --test-strings ",a,b,aa,ab" --verbose"""
    
    run_command(cmd2)


def demo_comparison():
    """Demonstrate comparing different model types."""
    print("\n" + "="*80)
    print("DEMO 3: Comparing different model approaches")
    print("="*80)
    
    # Extract DFA from the same concept using different approaches
    
    # 1. Function approach - starts with 'a'
    cmd1 = """python lstar_extract.py --model-type function --model-path examples/simple_classifier.py --function-name starts_with_a --alphabet "ab" --output examples/starts_a_function.png --test-strings ",a,b,aa,ab,ba,bb,aaa" --verbose"""
    
    run_command(cmd1)
    
    # 2. Function approach - binary divisible by 3
    cmd2 = """python lstar_extract.py --model-type function --model-path examples/simple_classifier.py --function-name binary_divisible_by_3 --alphabet "01" --output examples/div_by_3.png --test-strings "0,1,00,01,10,11,000,001,010,011,100,101,110,111" --verbose"""
    
    run_command(cmd2)


def demo_advanced_features():
    """Demonstrate advanced features of the tool."""
    print("\n" + "="*80)
    print("DEMO 4: Advanced features")
    print("="*80)
    
    # Example 1: Different alphabet formats
    cmd1 = """python lstar_extract.py --model-type function --model-path examples/simple_classifier.py --function-name palindrome_classifier --alphabet '["a", "b", "c"]' --output examples/palindrome_json_alphabet.png --test-strings ",a,aa,abc,aba,abcba" --verbose"""
    
    run_command(cmd1)
    
    # Example 2: Adjusting L* parameters
    cmd2 = """python lstar_extract.py --model-type function --model-path examples/simple_classifier.py --function-name even_length_classifier --alphabet "ab" --max-prefix-len 2 --max-suffix-len 3 --output examples/even_length_tuned.png --test-strings ",a,aa,aaa,aaaa" --verbose"""
    
    run_command(cmd2)


def demo_help():
    """Show the help message."""
    print("\n" + "="*80)
    print("DEMO: Help message")
    print("="*80)
    
    cmd = "python lstar_extract.py --help"
    run_command(cmd)


def main():
    """Run all demonstrations."""
    print("L* DFA Extraction Tool - Demonstration")
    print("This script demonstrates various uses of the lstar_extract.py tool")
    print("Generated files will be saved in the examples/ directory")
    
    # Create examples directory
    os.makedirs("examples", exist_ok=True)
    
    try:
        # Show help first
        demo_help()
        
        # Run demonstrations
        demo_function_extraction()
        demo_pytorch_extraction()
        demo_comparison()
        demo_advanced_features()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE!")
        print("="*80)
        print("Check the examples/ directory for generated DFA files:")
        print("- .png files: Visual representations of the learned DFAs")
        print("- .dot files: Graphviz DOT format files")
        print("\nYou can view PNG files directly or convert DOT files using:")
        print("  dot -Tpng input.dot -o output.png")
        
        # List generated files
        examples_dir = Path("examples")
        if examples_dir.exists():
            generated_files = list(examples_dir.glob("*.png")) + list(examples_dir.glob("*.dot"))
            if generated_files:
                print(f"\nGenerated files:")
                for file in sorted(generated_files):
                    print(f"  {file}")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 