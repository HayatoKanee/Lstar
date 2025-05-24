#!/usr/bin/env python3
"""
L* DFA Extraction Tool

Command-line tool to extract DFA models from any RNN using the L* active learning algorithm.
Supports PyTorch models, custom functions, and Hugging Face models.

Usage examples:
    # Extract DFA from PyTorch model
    python lstar_extract.py --model-path my_model.pth --model-type pytorch --model-class MyRNNClass --alphabet "01" --output learned_dfa.png

    # Extract from custom function
    python lstar_extract.py --model-path custom_classifier.py --model-type function --function-name my_classifier --alphabet "abc" --output result.dot

    # Extract from Hugging Face model
    python lstar_extract.py --model-type huggingface --model-name bert-base-uncased --alphabet "abcdefghijklmnopqrstuvwxyz " --output sentiment_dfa.png
"""

import argparse
import sys
import os
import importlib.util
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from rnn_adapters import (
    PyTorchRNNAdapter, 
    CustomFunctionRNNAdapter, 
    HuggingFaceRNNAdapter,
    RNNLoader
)
from lstar.oracles import AnalyzingGenericRNNMembershipOracle
from lstar import LStarLearner
from lstar.EquivalenceOracle import WMethodEquivalenceOracle


class LStarExtractor:
    """Main class for the L* extraction tool."""
    
    def __init__(self, args):
        self.args = args
        self.rnn = None
        self.alphabet = self._parse_alphabet(args.alphabet)
        
    def _parse_alphabet(self, alphabet_str: str) -> List[str]:
        """Parse alphabet from string specification."""
        if alphabet_str.startswith('[') and alphabet_str.endswith(']'):
            # JSON list format: ["a", "b", "c"]
            try:
                return json.loads(alphabet_str)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON alphabet format: {alphabet_str}")
        else:
            # Simple string format: "abc" -> ["a", "b", "c"]
            return list(alphabet_str)
    
    def _load_pytorch_model(self) -> PyTorchRNNAdapter:
        """Load PyTorch model from file."""
        if not self.args.model_class:
            raise ValueError("--model-class is required for PyTorch models")
        
        # Import the model class
        if self.args.model_module:
            # Load from specified module
            spec = importlib.util.spec_from_file_location("model_module", self.args.model_module)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model_class = getattr(module, self.args.model_class)
        else:
            # Try to import from current directory or common locations
            try:
                # Try importing as if it's a standard class
                parts = self.args.model_class.split('.')
                if len(parts) == 1:
                    # Assume it's in a file named after the class
                    module_name = parts[0].lower()
                    try:
                        module = importlib.import_module(module_name)
                        model_class = getattr(module, parts[0])
                    except ImportError:
                        raise ValueError(f"Could not import {self.args.model_class}. Use --model-module to specify the file.")
                else:
                    # Import from specified module
                    module = importlib.import_module('.'.join(parts[:-1]))
                    model_class = getattr(module, parts[-1])
            except (ImportError, AttributeError):
                raise ValueError(f"Could not import model class {self.args.model_class}")
        
        # Parse model kwargs
        model_kwargs = {}
        if self.args.model_kwargs:
            try:
                model_kwargs = json.loads(self.args.model_kwargs)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON for model kwargs: {self.args.model_kwargs}")
        
        # Load the model
        return RNNLoader.from_pytorch_file(
            model_path=self.args.model_path,
            model_class=model_class,
            alphabet=self.alphabet,
            model_kwargs=model_kwargs,
            threshold=self.args.threshold,
            device=self.args.device
        )
    
    def _load_function_model(self) -> CustomFunctionRNNAdapter:
        """Load custom function from Python file."""
        if not self.args.function_name:
            raise ValueError("--function-name is required for function models")
        
        # Load the Python file containing the function
        spec = importlib.util.spec_from_file_location("user_function", self.args.model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function
        try:
            prediction_function = getattr(module, self.args.function_name)
        except AttributeError:
            raise ValueError(f"Function {self.args.function_name} not found in {self.args.model_path}")
        
        return CustomFunctionRNNAdapter(
            prediction_function=prediction_function,
            alphabet=self.alphabet,
            threshold=self.args.threshold
        )
    
    def _load_huggingface_model(self) -> HuggingFaceRNNAdapter:
        """Load Hugging Face model."""
        model_name = self.args.model_name or self.args.model_path
        if not model_name:
            raise ValueError("--model-name or --model-path is required for Hugging Face models")
        
        return HuggingFaceRNNAdapter(
            model_name_or_path=model_name,
            alphabet=self.alphabet,
            threshold=self.args.threshold
        )
    
    def load_model(self):
        """Load the RNN model based on the specified type."""
        print(f"Loading {self.args.model_type} model...")
        
        if self.args.model_type == 'pytorch':
            self.rnn = self._load_pytorch_model()
        elif self.args.model_type == 'function':
            self.rnn = self._load_function_model()
        elif self.args.model_type == 'huggingface':
            self.rnn = self._load_huggingface_model()
        else:
            raise ValueError(f"Unsupported model type: {self.args.model_type}")
        
        print(f"Model loaded successfully. Alphabet: {self.alphabet}")
    
    def test_model(self):
        """Test the model with sample strings."""
        if not self.args.test_strings:
            return
        
        print("\nTesting model with sample strings:")
        test_strings = self.args.test_strings.split(',')
        
        for test_str in test_strings:
            test_str = test_str.strip()
            try:
                result = self.rnn.query(test_str)
                confidence = self.rnn.get_confidence(test_str)
                print(f"  '{test_str}': {result} (confidence: {confidence:.3f})")
            except Exception as e:
                print(f"  '{test_str}': Error - {e}")
    
    def extract_dfa(self):
        """Extract DFA using L* algorithm."""
        print(f"\nExtracting DFA using L* algorithm...")
        
        # Create oracles
        if self.args.verbose:
            membership_oracle = AnalyzingGenericRNNMembershipOracle(self.rnn)
        else:
            from lstar.oracles import GenericRNNMembershipOracle
            membership_oracle = GenericRNNMembershipOracle(self.rnn)
        
        equivalence_oracle = WMethodEquivalenceOracle(
            membership_oracle, 
            self.alphabet,
            max_prefix_len=self.args.max_prefix_len,
            max_suffix_len=self.args.max_suffix_len
        )
        
        # Create learner and run
        learner = LStarLearner(self.alphabet, membership_oracle, equivalence_oracle)
        
        start_time = time.time()
        learned_dfa = learner.run()
        end_time = time.time()
        
        # Print results
        print(f"DFA extracted successfully!")
        print(f"  States: {len(learned_dfa.states)}")
        print(f"  Learning time: {end_time - start_time:.3f} seconds")
        print(f"  Queries made: {membership_oracle.get_query_count()}")
        
        if self.args.verbose and hasattr(membership_oracle, 'print_query_statistics'):
            membership_oracle.print_query_statistics()
            membership_oracle.analyze_patterns()
        
        return learned_dfa
    
    def save_dfa(self, dfa):
        """Save DFA to specified output format."""
        output_path = Path(self.args.output)
        
        if output_path.suffix.lower() == '.png':
            print(f"Saving DFA visualization to {output_path}")
            dfa.write_png(str(output_path.with_suffix('')))  # DFA.write_png adds .png
        elif output_path.suffix.lower() == '.dot':
            print(f"Saving DFA DOT file to {output_path}")
            dfa.write_dot(str(output_path.with_suffix('')))  # DFA.write_dot adds .dot
        else:
            # Default to PNG
            print(f"No extension specified, saving as PNG to {output_path.with_suffix('.png')}")
            dfa.write_png(str(output_path.with_suffix('')))
    
    def run(self):
        """Run the complete extraction process."""
        try:
            self.load_model()
            self.test_model()
            dfa = self.extract_dfa()
            self.save_dfa(dfa)
            print(f"\nDFA extraction completed successfully!")
            
        except Exception as e:
            print(f"Error: {e}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract DFA models from RNNs using L* active learning algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PyTorch model
  python lstar_extract.py --model-path model.pth --model-type pytorch --model-class MyRNN --alphabet "01" --output dfa.png

  # Custom function
  python lstar_extract.py --model-path classifier.py --model-type function --function-name my_func --alphabet "abc" --output dfa.dot

  # Hugging Face model
  python lstar_extract.py --model-type huggingface --model-name bert-base-uncased --alphabet "abcdefghijklmnopqrstuvwxyz " --output dfa.png

  # With model parameters
  python lstar_extract.py --model-path model.pth --model-type pytorch --model-class RNN --model-kwargs '{"hidden_size": 64, "num_layers": 2}' --alphabet "01" --output result.png
        """
    )
    
    # Model specification
    model_group = parser.add_argument_group('Model Specification')
    model_group.add_argument('--model-type', required=True, 
                           choices=['pytorch', 'function', 'huggingface'],
                           help='Type of model to load')
    model_group.add_argument('--model-path',
                           help='Path to model file (.pth for PyTorch, .py for functions)')
    model_group.add_argument('--model-name',
                           help='Model name (for Hugging Face models)')
    
    # PyTorch specific
    pytorch_group = parser.add_argument_group('PyTorch Model Options')
    pytorch_group.add_argument('--model-class',
                             help='Name of the model class (required for PyTorch)')
    pytorch_group.add_argument('--model-module',
                             help='Path to Python file containing model class')
    pytorch_group.add_argument('--model-kwargs',
                             help='JSON string of model constructor arguments')
    
    # Function specific
    function_group = parser.add_argument_group('Function Model Options')
    function_group.add_argument('--function-name',
                              help='Name of the prediction function (required for functions)')
    
    # General options
    general_group = parser.add_argument_group('General Options')
    general_group.add_argument('--alphabet', required=True,
                             help='Alphabet as string ("abc") or JSON list (["a","b","c"])')
    general_group.add_argument('--threshold', type=float, default=0.5,
                             help='Classification threshold (default: 0.5)')
    general_group.add_argument('--device', default='cpu',
                             help='Device for PyTorch models (default: cpu)')
    
    # L* algorithm options
    lstar_group = parser.add_argument_group('L* Algorithm Options')
    lstar_group.add_argument('--max-prefix-len', type=int, default=3,
                           help='Maximum prefix length for equivalence queries (default: 3)')
    lstar_group.add_argument('--max-suffix-len', type=int, default=4,
                           help='Maximum suffix length for equivalence queries (default: 4)')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', '-o', required=True,
                            help='Output file path (.png or .dot)')
    output_group.add_argument('--test-strings',
                            help='Comma-separated test strings to evaluate model')
    output_group.add_argument('--verbose', '-v', action='store_true',
                            help='Enable verbose output and detailed statistics')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_type == 'pytorch' and not args.model_class:
        parser.error("--model-class is required for PyTorch models")
    
    if args.model_type == 'function' and not args.function_name:
        parser.error("--function-name is required for function models")
    
    if args.model_type in ['pytorch', 'function'] and not args.model_path:
        parser.error(f"--model-path is required for {args.model_type} models")
    
    if args.model_type == 'huggingface' and not (args.model_name or args.model_path):
        parser.error("--model-name or --model-path is required for Hugging Face models")
    
    # Create and run extractor
    extractor = LStarExtractor(args)
    extractor.run()


if __name__ == "__main__":
    main() 