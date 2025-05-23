# L* Algorithm with RNN Integration

This branch demonstrates how to use the L* active learning algorithm to extract DFA models from trained Recurrent Neural Networks (RNNs). This is particularly useful for understanding what language patterns an RNN has learned and for formal verification purposes.

## Overview

The L* algorithm can learn regular languages by querying a membership oracle and an equivalence oracle. In this implementation, we've extended the framework to work with RNNs as black-box models, enabling automatic extraction of finite automata that approximate the language recognized by the neural network.

## New Components

### 1. DummyRNN (`DummyRNN.py`)

A simple RNN implementation that can be trained on regular language patterns:

- **SimpleRNN**: PyTorch-based LSTM network for binary string classification
- **DummyRNN**: Wrapper class providing training and querying functionality
- Supports training on regex patterns and querying individual strings

### 2. RNN Membership Oracle (`RNNMembershipOracle.py`)

Adapts trained RNNs to work with the L* framework:

- **RNNMembershipOracle**: Basic oracle that queries the RNN
- **AnalyzingRNNMembershipOracle**: Extended version with logging and statistics

### 3. Complete Example (`rnn_example.py`)

Demonstrates the full pipeline:
1. Train RNN on a pattern (e.g., "contains substring '01'")
2. Use L* algorithm to learn DFA from RNN
3. Compare learned DFA with original pattern
4. Analyze query statistics and model accuracy

## Usage

### Basic Example

```python
from DummyRNN import DummyRNN
from RNNMembershipOracle import RNNMembershipOracle
from EquivalenceOracle import WMethodEquivalenceOracle
from LStarLearner import LStarLearner

# Train RNN on a pattern
alphabet = ["0", "1"]
pattern = r'^(0|1)*01(0|1)*$'  # Contains "01"
rnn = DummyRNN(alphabet, pattern)

# Set up L* framework
membership_oracle = RNNMembershipOracle(rnn)
equivalence_oracle = WMethodEquivalenceOracle(membership_oracle, alphabet)
learner = LStarLearner(alphabet, membership_oracle, equivalence_oracle)

# Learn DFA from RNN
learned_dfa = learner.run()
```

### Running the Complete Example

```bash
python rnn_example.py
```

This will:
- Train an RNN on the pattern "binary strings containing '01'"
- Use L* to extract a DFA from the trained RNN
- Compare the learned DFA with the original pattern
- Display statistics about the learning process

## Key Features

### RNN Training
- Automatic training data generation from regex patterns
- Configurable training parameters (epochs, hidden size, etc.)
- Support for binary classification of strings

### L* Integration
- RNN queries are seamlessly integrated with existing L* framework
- W-method equivalence checking for robust DFA extraction
- Query logging and analysis capabilities

### Analysis Tools
- Model comparison between learned DFA and original pattern
- Query statistics (count, accuracy, confidence scores)
- Visualization of learned DFA structure

## Expected Output

When running the example, you should see:

1. **RNN Training**: Progress of neural network training on the target pattern
2. **L* Execution**: Step-by-step learning process with hypothesis generation
3. **Query Statistics**: Number of queries, acceptance rates, confidence scores
4. **Model Comparison**: Accuracy of learned DFA vs. original pattern
5. **Detailed Tests**: Per-string comparison between DFA, RNN, and original pattern

## Technical Details

### Pattern Recognition
The dummy RNN is designed to learn simple regular patterns like:
- Substring containment (e.g., contains "01")
- Prefix/suffix patterns
- Length constraints
- Alternation patterns

### Query Efficiency
The W-method equivalence oracle is configured to balance thoroughness with efficiency:
- `max_prefix_len=3`: Tests transition sequences up to length 3
- `max_suffix_len=4`: Uses distinguishing suffixes up to length 4
- Typically requires 50-200 queries for simple patterns

### Accuracy Considerations
- RNN training quality affects the learnability of patterns
- Some patterns may require more training data or different architectures
- The L* algorithm will learn the best DFA approximation of the RNN's behavior

## Future Extensions

This framework can be extended to:
- More complex RNN architectures (GRU, Transformer)
- Multi-class classification (beyond binary acceptance)
- Real-world trained RNNs (sentiment analysis, language models)
- Different alphabet sizes and string types
- Online learning scenarios where the RNN continues to train

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch (neural network implementation)
- NumPy (numerical operations)
- Graphviz (DFA visualization)
- Matplotlib (plotting and visualization) 