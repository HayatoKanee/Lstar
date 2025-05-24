"""
L* CLI: Extract DFAs from Any RNN

A powerful command-line tool that uses the L* active learning algorithm to extract 
interpretable Deterministic Finite Automata (DFA) from any type of Recurrent Neural Network.

## Primary Tool

The main deliverable is the CLI tool:
    python lstar_cli/lstar_extract.py --help

This tool can extract DFAs from:
- PyTorch models (.pth files)
- Python functions (any classifier function) 
- Hugging Face transformers
- Any custom RNN implementation

## Quick Start

    # Extract DFA from a PyTorch model
    python lstar_cli/lstar_extract.py \
      --model-type pytorch \
      --model-path your_model.pth \
      --model-class YourRNNClass \
      --alphabet "01" \
      --output learned_dfa.png

    # Extract DFA from any Python function
    python lstar_cli/lstar_extract.py \
      --model-type function \
      --model-path classifier.py \
      --function-name my_classifier \
      --alphabet "abc" \
      --output dfa.png

## Programmatic Access

All CLI functionality is also available programmatically:
"""

__version__ = "1.0.0"
__author__ = "Paul"

# Core algorithm components (used by CLI tool)
from lstar import DFA, ObservationTable, LStarLearner, EquivalenceOracle

# Oracle implementations (used by CLI tool)
from lstar.oracles import (
    MembershipOracle, 
    RegexMembershipOracle,
    RNNMembershipOracle, 
    GenericRNNMembershipOracle,
    AnalyzingGenericRNNMembershipOracle
)

# RNN adapters (core of CLI tool functionality)
from rnn_adapters import (
    GenericRNNInterface,
    PyTorchRNNAdapter,
    HuggingFaceRNNAdapter, 
    CustomFunctionRNNAdapter,
    RNNLoader,
    DummyRNN
)

__all__ = [
    # Core L* components
    'DFA', 'ObservationTable', 'LStarLearner', 'EquivalenceOracle',
    
    # Oracles
    'MembershipOracle', 'RegexMembershipOracle', 'RNNMembershipOracle', 
    'GenericRNNMembershipOracle', 'AnalyzingGenericRNNMembershipOracle',
    
    # RNN adapters (main CLI functionality)
    'GenericRNNInterface', 'PyTorchRNNAdapter', 'HuggingFaceRNNAdapter',
    'CustomFunctionRNNAdapter', 'RNNLoader', 'DummyRNN'
]

# CLI Tool Location
CLI_TOOL = "lstar_cli/lstar_extract.py" 