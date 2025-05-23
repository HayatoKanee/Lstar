import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict
import re
from collections import defaultdict


class SimpleRNN(nn.Module):
    """
    A simple RNN that can be trained to recognize string patterns.
    This serves as a dummy RNN for testing the L* algorithm.
    """
    
    def __init__(self, vocab_size: int, hidden_size: int = 32, num_layers: int = 1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer for characters
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RNN layer
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer for binary classification
        self.classifier = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward through RNN
        out, _ = self.rnn(embedded, (h0, c0))
        
        # Use the last output for classification
        last_output = out[:, -1, :]  # (batch_size, hidden_size)
        
        # Classification
        logits = self.classifier(last_output)
        return self.sigmoid(logits).squeeze()


class DummyRNN:
    """
    A wrapper class for the RNN that provides easy training and querying functionality.
    """
    
    def __init__(self, alphabet: List[str], pattern: str = None):
        self.alphabet = alphabet
        self.vocab_size = len(alphabet) + 2  # +2 for padding and unknown tokens
        
        # Create character to index mapping
        self.char_to_idx = {char: i+1 for i, char in enumerate(alphabet)}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<UNK>'] = len(alphabet) + 1
        
        # Initialize the RNN
        self.model = SimpleRNN(self.vocab_size)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # If a pattern is provided, train on it
        if pattern:
            self.train_on_pattern(pattern)
    
    def string_to_tensor(self, string: str) -> torch.Tensor:
        """Convert a string to a tensor of character indices."""
        if not string:  # Empty string
            return torch.tensor([[0]], dtype=torch.long)
        
        indices = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in string]
        return torch.tensor([indices], dtype=torch.long)
    
    def generate_training_data(self, pattern: str, num_positive: int = 200, num_negative: int = 200) -> Tuple[List[str], List[bool]]:
        """
        Generate training data based on a regex pattern.
        Returns lists of strings and their corresponding labels.
        """
        regex = re.compile(pattern)
        strings = []
        labels = []
        
        # Generate positive examples
        positive_count = 0
        max_attempts = 1000
        attempt = 0
        
        while positive_count < num_positive and attempt < max_attempts:
            # Generate random strings of various lengths
            length = np.random.randint(1, 8)
            string = ''.join(np.random.choice(self.alphabet, length))
            
            if regex.fullmatch(string):
                strings.append(string)
                labels.append(True)
                positive_count += 1
            
            attempt += 1
        
        # If we couldn't generate enough from random sampling, create some manually
        if positive_count < num_positive:
            # For simple patterns like "contains 01", manually create examples
            for i in range(num_positive - positive_count):
                # Create strings that definitely match the pattern
                prefix_len = np.random.randint(0, 3)
                suffix_len = np.random.randint(0, 3)
                prefix = ''.join(np.random.choice(self.alphabet, prefix_len))
                suffix = ''.join(np.random.choice(self.alphabet, suffix_len))
                
                # Insert a pattern that we know matches (this is pattern-specific)
                if "01" in pattern:
                    middle = "01"
                elif "10" in pattern:
                    middle = "10"
                else:
                    middle = self.alphabet[0]  # fallback
                
                candidate = prefix + middle + suffix
                if regex.fullmatch(candidate):
                    strings.append(candidate)
                    labels.append(True)
        
        # Generate negative examples
        negative_count = 0
        attempt = 0
        
        while negative_count < num_negative and attempt < max_attempts:
            length = np.random.randint(1, 8)
            string = ''.join(np.random.choice(self.alphabet, length))
            
            if not regex.fullmatch(string):
                strings.append(string)
                labels.append(False)
                negative_count += 1
            
            attempt += 1
        
        return strings, labels
    
    def train_on_pattern(self, pattern: str, epochs: int = 100):
        """Train the RNN to recognize the given regex pattern."""
        print(f"Training RNN on pattern: {pattern}")
        
        # Generate training data
        strings, labels = self.generate_training_data(pattern)
        
        print(f"Generated {len(strings)} training examples")
        print(f"Positive examples: {sum(labels)}, Negative examples: {len(labels) - sum(labels)}")
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            
            # Shuffle the data
            combined = list(zip(strings, labels))
            np.random.shuffle(combined)
            strings_shuffled, labels_shuffled = zip(*combined)
            
            for string, label in zip(strings_shuffled, labels_shuffled):
                self.optimizer.zero_grad()
                
                # Convert to tensors - fix tensor dimension issue
                input_tensor = self.string_to_tensor(string)
                target = torch.tensor(float(label), dtype=torch.float32)
                
                # Forward pass
                output = self.model(input_tensor)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                predicted = (output > 0.5).float()
                correct += (predicted == target).float().item()
            
            if epoch % 20 == 0:
                accuracy = correct / len(strings)
                avg_loss = total_loss / len(strings)
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        print("Training completed!")
    
    def query(self, string: str) -> bool:
        """
        Query the RNN with a string and return whether it accepts or rejects.
        This is the interface used by the MembershipOracle.
        """
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.string_to_tensor(string)
            output = self.model(input_tensor)
            return bool(output.item() > 0.5)
    
    def get_confidence(self, string: str) -> float:
        """Get the confidence score for a string (between 0 and 1)."""
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.string_to_tensor(string)
            output = self.model(input_tensor)
            return float(output.item())


# Example usage and testing
if __name__ == "__main__":
    # Test the dummy RNN
    alphabet = ["0", "1"]
    pattern = r'^(0|1)*01(0|1)*$'  # Contains substring "01"
    
    rnn = DummyRNN(alphabet, pattern)
    
    # Test some strings
    test_strings = ["01", "001", "1001", "0110", "00", "11", "10", ""]
    print("\nTesting RNN predictions:")
    for s in test_strings:
        prediction = rnn.query(s)
        confidence = rnn.get_confidence(s)
        # Verify with actual regex
        actual = bool(re.fullmatch(pattern, s))
        print(f"'{s}': RNN={prediction} (conf={confidence:.3f}), Actual={actual}") 