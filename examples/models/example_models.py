#!/usr/bin/env python3
"""
Example PyTorch model classes for use with lstar_extract.py

These models can be referenced with the command-line tool:
    python lstar_extract.py --model-path my_trained_model.pth --model-type pytorch --model-class SimpleRNN --model-module examples/example_models.py --alphabet "abc" --output result.png
"""

import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    """Simple RNN for binary sequence classification."""
    
    def __init__(self, vocab_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        
        # RNN output
        output, hidden = self.rnn(embedded)  # output: (batch_size, seq_len, hidden_size)
        
        # Use the last hidden state for classification
        last_hidden = hidden[-1]  # (batch_size, hidden_size)
        logits = self.classifier(last_hidden)  # (batch_size, 1)
        
        return torch.sigmoid(logits.squeeze(-1))  # (batch_size,)


class SimpleLSTM(nn.Module):
    """Simple LSTM for binary sequence classification."""
    
    def __init__(self, vocab_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        
        # LSTM output
        output, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state for classification
        last_hidden = hidden[-1]  # (batch_size, hidden_size)
        last_hidden = self.dropout(last_hidden)
        
        logits = self.classifier(last_hidden)  # (batch_size, 1)
        return torch.sigmoid(logits.squeeze(-1))  # (batch_size,)


class SimpleGRU(nn.Module):
    """Simple GRU for binary sequence classification."""
    
    def __init__(self, vocab_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        
        # GRU output
        output, hidden = self.gru(embedded)
        
        # Use the last hidden state for classification
        last_hidden = hidden[-1]  # (batch_size, hidden_size)
        last_hidden = self.dropout(last_hidden)
        
        logits = self.classifier(last_hidden)  # (batch_size, 1)
        return torch.sigmoid(logits.squeeze(-1))  # (batch_size,)


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM for sequence classification."""
    
    def __init__(self, vocab_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # *2 because bidirectional
        self.classifier = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        
        # Bidirectional LSTM output
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate forward and backward hidden states
        # hidden shape: (num_layers*2, batch_size, hidden_size)
        forward_hidden = hidden[-2]  # Last layer forward
        backward_hidden = hidden[-1]  # Last layer backward
        
        concat_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        concat_hidden = self.dropout(concat_hidden)
        
        logits = self.classifier(concat_hidden)  # (batch_size, 1)
        return torch.sigmoid(logits.squeeze(-1))  # (batch_size,)


class AttentionRNN(nn.Module):
    """RNN with simple attention mechanism."""
    
    def __init__(self, vocab_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Attention layer
        self.attention = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        
        # LSTM output
        output, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_size)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(output), dim=1)  # (batch_size, seq_len, 1)
        attended_output = torch.sum(attention_weights * output, dim=1)  # (batch_size, hidden_size)
        
        logits = self.classifier(attended_output)  # (batch_size, 1)
        return torch.sigmoid(logits.squeeze(-1))  # (batch_size,)


class ConvolutionalRNN(nn.Module):
    """CNN + RNN hybrid model."""
    
    def __init__(self, vocab_size, embed_size=64, hidden_size=64, num_filters=100, filter_sizes=[3, 4, 5]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_size, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # RNN layer
        conv_output_size = len(filter_sizes) * num_filters
        self.lstm = nn.LSTM(conv_output_size, hidden_size, batch_first=True)
        
        self.classifier = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_size)
        embedded = embedded.transpose(1, 2)  # (batch_size, embed_size, seq_len)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # (batch_size, num_filters, new_seq_len)
            pooled = torch.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        # Concatenate conv outputs
        conv_concat = torch.cat(conv_outputs, dim=1)  # (batch_size, total_filters)
        conv_concat = self.dropout(conv_concat)
        
        # Add sequence dimension for LSTM
        conv_concat = conv_concat.unsqueeze(1)  # (batch_size, 1, total_filters)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(conv_concat)
        
        # Classification
        logits = self.classifier(hidden[-1])  # (batch_size, 1)
        return torch.sigmoid(logits.squeeze(-1))  # (batch_size,)


class SimpleClassifier(nn.Module):
    """Very simple classifier for testing purposes."""
    
    def __init__(self, vocab_size, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        output = self.classifier(hidden[-1])
        return torch.sigmoid(output.squeeze(-1))


# Alias for backward compatibility
MyRNN = SimpleRNN 