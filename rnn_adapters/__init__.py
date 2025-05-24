"""
RNN Adapters for L* Algorithm

This package contains adapters for different RNN frameworks:
- Generic RNN interface
- PyTorch RNN adapter
- Hugging Face RNN adapter
- Custom function adapter
- DummyRNN implementation
"""

from .GenericRNN import (
    GenericRNNInterface,
    PyTorchRNNAdapter,
    HuggingFaceRNNAdapter,
    CustomFunctionRNNAdapter,
    RNNLoader
)
from .DummyRNN import DummyRNN

__all__ = [
    'GenericRNNInterface',
    'PyTorchRNNAdapter', 
    'HuggingFaceRNNAdapter',
    'CustomFunctionRNNAdapter',
    'RNNLoader',
    'DummyRNN'
] 