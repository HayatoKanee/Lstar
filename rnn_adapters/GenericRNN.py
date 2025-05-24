from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict, Optional, Callable
import torch
import torch.nn as nn
import numpy as np
import re


class GenericRNNInterface(ABC):
    """
    Abstract interface that any RNN must implement to work with the L* algorithm.
    This allows the framework to work with RNNs from any source or framework.
    """
    
    @abstractmethod
    def query(self, string: str) -> bool:
        """
        Query the RNN with a string and return binary classification result.
        
        Args:
            string: Input string to classify
            
        Returns:
            bool: True if the string is accepted/positive, False otherwise
        """
        pass
    
    @abstractmethod
    def get_confidence(self, string: str) -> float:
        """
        Get the confidence score for a string classification.
        
        Args:
            string: Input string to classify
            
        Returns:
            float: Confidence score between 0 and 1
        """
        pass
    
    @abstractmethod
    def get_alphabet(self) -> List[str]:
        """
        Get the alphabet (vocabulary) that this RNN can handle.
        
        Returns:
            List[str]: List of characters/tokens the RNN can process
        """
        pass


class PyTorchRNNAdapter(GenericRNNInterface):
    """
    Adapter for PyTorch RNN models to work with the L* framework.
    Handles various PyTorch model architectures and preprocessing.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 alphabet: List[str],
                 tokenizer: Optional[Callable[[str], List[int]]] = None,
                 threshold: float = 0.5,
                 device: str = 'cpu'):
        """
        Initialize the PyTorch RNN adapter.
        
        Args:
            model: PyTorch model that outputs logits or probabilities
            alphabet: List of characters/tokens the model can handle
            tokenizer: Function to convert strings to token indices (if None, uses char-to-index)
            threshold: Classification threshold for binary decision
            device: Device to run the model on
        """
        self.model = model
        self.alphabet = alphabet
        self.threshold = threshold
        self.device = device
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
        
        # Setup tokenizer
        if tokenizer is None:
            # Default character-level tokenizer
            self.char_to_idx = {char: i for i, char in enumerate(alphabet)}
            self.tokenizer = self._default_tokenizer
        else:
            self.tokenizer = tokenizer
    
    def _default_tokenizer(self, string: str) -> List[int]:
        """Default character-level tokenization."""
        return [self.char_to_idx.get(char, 0) for char in string]
    
    def _string_to_tensor(self, string: str) -> torch.Tensor:
        """Convert string to model input tensor."""
        if not string:
            # Handle empty string
            tokens = [0]  # Use padding token
        else:
            tokens = self.tokenizer(string)
        
        return torch.tensor([tokens], dtype=torch.long).to(self.device)
    
    def _get_model_output(self, string: str) -> float:
        """Get raw model output for a string."""
        with torch.no_grad():
            input_tensor = self._string_to_tensor(string)
            output = self.model(input_tensor)
            
            # Handle different output formats
            if output.dim() > 1:
                output = output.squeeze()
            
            # Convert to probability if needed
            if hasattr(self.model, 'sigmoid') or 'sigmoid' in str(self.model).lower():
                # Assume output is already probability
                prob = float(output.item())
            else:
                # Apply sigmoid to convert logits to probability
                prob = float(torch.sigmoid(output).item())
            
            return prob
    
    def query(self, string: str) -> bool:
        """Query the RNN for binary classification."""
        prob = self._get_model_output(string)
        return prob > self.threshold
    
    def get_confidence(self, string: str) -> float:
        """Get confidence score (probability) for the string."""
        return self._get_model_output(string)
    
    def get_alphabet(self) -> List[str]:
        """Return the alphabet this RNN can handle."""
        return self.alphabet.copy()


class HuggingFaceRNNAdapter(GenericRNNInterface):
    """
    Adapter for Hugging Face transformer models to work with the L* framework.
    """
    
    def __init__(self, 
                 model_name_or_path: str,
                 alphabet: List[str],
                 threshold: float = 0.5):
        """
        Initialize the Hugging Face model adapter.
        
        Args:
            model_name_or_path: Path to model or model name from Hugging Face Hub
            alphabet: List of characters the model should handle
            threshold: Classification threshold
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
            self.model.eval()
        except ImportError:
            raise ImportError("transformers library required for HuggingFace adapter. Install with: pip install transformers")
        
        self.alphabet = alphabet
        self.threshold = threshold
    
    def _get_model_output(self, string: str) -> float:
        """Get model prediction for a string."""
        inputs = self.tokenizer(string, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Convert to probability (assuming binary classification)
            if logits.shape[-1] == 1:
                # Single output neuron
                prob = torch.sigmoid(logits).item()
            else:
                # Multiple classes, take positive class probability
                probs = torch.softmax(logits, dim=-1)
                prob = probs[0, 1].item() if logits.shape[-1] > 1 else probs[0, 0].item()
        
        return prob
    
    def query(self, string: str) -> bool:
        """Query the model for binary classification."""
        prob = self._get_model_output(string)
        return prob > self.threshold
    
    def get_confidence(self, string: str) -> float:
        """Get confidence score for the string."""
        return self._get_model_output(string)
    
    def get_alphabet(self) -> List[str]:
        """Return the alphabet."""
        return self.alphabet.copy()


class CustomFunctionRNNAdapter(GenericRNNInterface):
    """
    Adapter that wraps any custom function to work with the L* framework.
    Useful for models from other frameworks or custom implementations.
    """
    
    def __init__(self,
                 prediction_function: Callable[[str], float],
                 alphabet: List[str],
                 threshold: float = 0.5):
        """
        Initialize with a custom prediction function.
        
        Args:
            prediction_function: Function that takes a string and returns a probability [0, 1]
            alphabet: List of characters the function can handle
            threshold: Classification threshold
        """
        self.prediction_function = prediction_function
        self.alphabet = alphabet
        self.threshold = threshold
    
    def query(self, string: str) -> bool:
        """Query using the custom function."""
        prob = self.prediction_function(string)
        return prob > self.threshold
    
    def get_confidence(self, string: str) -> float:
        """Get confidence from the custom function."""
        return self.prediction_function(string)
    
    def get_alphabet(self) -> List[str]:
        """Return the alphabet."""
        return self.alphabet.copy()


class RNNLoader:
    """
    Utility class to help users load and adapt various types of RNNs.
    """
    
    @staticmethod
    def from_pytorch_file(model_path: str, 
                         model_class: type,
                         alphabet: List[str],
                         model_kwargs: Dict[str, Any] = None,
                         **adapter_kwargs) -> PyTorchRNNAdapter:
        """
        Load a PyTorch model from file.
        
        Args:
            model_path: Path to the saved model (.pth or .pt file)
            model_class: The model class to instantiate
            alphabet: Alphabet for the model
            model_kwargs: Arguments to pass to model constructor
            **adapter_kwargs: Additional arguments for the adapter
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # Load the model
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        return PyTorchRNNAdapter(model, alphabet, **adapter_kwargs)
    
    @staticmethod
    def from_huggingface(model_name: str,
                        alphabet: List[str],
                        **adapter_kwargs) -> HuggingFaceRNNAdapter:
        """
        Load a model from Hugging Face Hub.
        
        Args:
            model_name: Model name or path
            alphabet: Alphabet for the model
            **adapter_kwargs: Additional arguments for the adapter
        """
        return HuggingFaceRNNAdapter(model_name, alphabet, **adapter_kwargs)
    
    @staticmethod
    def from_function(prediction_function: Callable[[str], float],
                     alphabet: List[str],
                     **adapter_kwargs) -> CustomFunctionRNNAdapter:
        """
        Create adapter from a custom prediction function.
        
        Args:
            prediction_function: Function that predicts probability for strings
            alphabet: Alphabet for the function
            **adapter_kwargs: Additional arguments for the adapter
        """
        return CustomFunctionRNNAdapter(prediction_function, alphabet, **adapter_kwargs)
    
    @staticmethod
    def from_dummy_rnn(dummy_rnn) -> GenericRNNInterface:
        """
        Adapt our DummyRNN to the generic interface.
        
        Args:
            dummy_rnn: Instance of DummyRNN
        """
        class DummyRNNAdapter(GenericRNNInterface):
            def __init__(self, rnn):
                self.rnn = rnn
            
            def query(self, string: str) -> bool:
                return self.rnn.query(string)
            
            def get_confidence(self, string: str) -> float:
                return self.rnn.get_confidence(string)
            
            def get_alphabet(self) -> List[str]:
                return self.rnn.alphabet
        
        return DummyRNNAdapter(dummy_rnn)


# Example usage and testing functions
def create_example_models():
    """Create some example models for testing."""
    
    # Example 1: Simple PyTorch model
    class SimpleClassifier(nn.Module):
        def __init__(self, vocab_size, hidden_size=32):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.classifier = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            embedded = self.embedding(x)
            _, (hidden, _) = self.rnn(embedded)
            output = self.classifier(hidden[-1])
            return torch.sigmoid(output)
    
    # Example 2: Custom function that recognizes palindromes
    def palindrome_classifier(string: str) -> float:
        """Simple function that returns high confidence for palindromes."""
        if len(string) <= 1:
            return 0.9
        is_palindrome = string == string[::-1]
        return 0.8 if is_palindrome else 0.2
    
    alphabet = ["a", "b", "c"]
    
    # Create models
    models = {
        "pytorch_model": PyTorchRNNAdapter(
            SimpleClassifier(len(alphabet) + 1), 
            alphabet
        ),
        "palindrome_function": CustomFunctionRNNAdapter(
            palindrome_classifier,
            alphabet
        )
    }
    
    return models


if __name__ == "__main__":
    # Test the adapters
    print("=== Testing Generic RNN Adapters ===\n")
    
    # Create example models
    models = create_example_models()
    
    # Test strings
    test_strings = ["", "a", "aa", "ab", "aba", "abc", "cbc"]
    
    for model_name, model in models.items():
        print(f"Testing {model_name}:")
        print(f"Alphabet: {model.get_alphabet()}")
        
        for string in test_strings:
            result = model.query(string)
            confidence = model.get_confidence(string)
            print(f"  '{string}': {result} (confidence: {confidence:.3f})")
        
        print() 