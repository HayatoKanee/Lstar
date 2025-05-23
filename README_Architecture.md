# L* Algorithm with Abstract Membership Oracle Architecture

This project implements the L* active learning algorithm with a flexible, extensible architecture for membership oracles. The framework now supports multiple types of membership oracles through an abstract base class design.

## Architecture Overview

### Abstract Base Class Design

The membership oracle system now follows a clean object-oriented design:

```
MembershipOracle (ABC)
├── RegexMembershipOracle
└── RNNMembershipOracle
    └── AnalyzingRNNMembershipOracle
```

### Core Components

#### 1. MembershipOracle (Abstract Base Class)
- **File**: `MembershipOracle.py`
- **Purpose**: Defines the interface that all membership oracles must implement
- **Key Method**: `query(word: str) -> bool`

#### 2. RegexMembershipOracle
- **File**: `MembershipOracle.py`
- **Purpose**: Uses regular expressions to define target languages
- **Use Case**: Learning from formally specified patterns

#### 3. RNNMembershipOracle
- **File**: `RNNMembershipOracle.py`
- **Purpose**: Uses trained RNNs as membership oracles
- **Use Case**: Learning from neural network models

#### 4. AnalyzingRNNMembershipOracle
- **File**: `RNNMembershipOracle.py`
- **Purpose**: Extended RNN oracle with query logging and statistics
- **Use Case**: Detailed analysis of the learning process

## Usage Examples

### Basic Regex Oracle
```python
from MembershipOracle import RegexMembershipOracle
from EquivalenceOracle import WMethodEquivalenceOracle
from LStarLearner import LStarLearner

# Define target language with regex
pattern = r'^(0|1)*01(0|1)*$'  # Contains "01"
alphabet = ["0", "1"]

# Create oracles
membership_oracle = RegexMembershipOracle(pattern)
equivalence_oracle = WMethodEquivalenceOracle(membership_oracle, alphabet)

# Learn DFA
learner = LStarLearner(alphabet, membership_oracle, equivalence_oracle)
dfa = learner.run()
```

### RNN Oracle
```python
from DummyRNN import DummyRNN
from RNNMembershipOracle import RNNMembershipOracle
from EquivalenceOracle import WMethodEquivalenceOracle
from LStarLearner import LStarLearner

# Train RNN on target pattern
rnn = DummyRNN(alphabet, pattern)

# Create oracles
membership_oracle = RNNMembershipOracle(rnn)
equivalence_oracle = WMethodEquivalenceOracle(membership_oracle, alphabet)

# Learn DFA
learner = LStarLearner(alphabet, membership_oracle, equivalence_oracle)
dfa = learner.run()
```

### Analyzing RNN Oracle
```python
from RNNMembershipOracle import AnalyzingRNNMembershipOracle

# Create analyzing oracle for detailed statistics
membership_oracle = AnalyzingRNNMembershipOracle(rnn)
equivalence_oracle = WMethodEquivalenceOracle(membership_oracle, alphabet)

learner = LStarLearner(alphabet, membership_oracle, equivalence_oracle)
dfa = learner.run()

# Print detailed query statistics
membership_oracle.print_query_statistics()
print(f"Total queries: {membership_oracle.get_query_count()}")
```

## Available Examples

### 1. Basic Example (`example.py`)
- Demonstrates L* with `RegexMembershipOracle`
- Simple pattern: strings containing "01"
- Shows basic DFA learning workflow

### 2. RNN Example (`rnn_example.py`)
- Comprehensive RNN-based learning demonstration
- Includes RNN training, L* learning, and validation
- Detailed analysis and comparison with original pattern

### 3. Comparison Example (`comparison_example.py`)
- **NEW**: Side-by-side comparison of both oracle types
- Shows both oracles learning the same language
- Validates that both approaches produce equivalent DFAs
- Demonstrates the flexibility of the abstract architecture

## Key Benefits of the New Architecture

### 1. **Extensibility**
- Easy to add new membership oracle types
- All oracles work seamlessly with existing L* framework
- Clean separation of concerns

### 2. **Consistency**
- Uniform interface across all oracle types
- Same L* algorithm works with any oracle implementation
- Predictable behavior and API

### 3. **Flexibility**
- Switch between oracle types without changing L* code
- Compare different approaches on same target language
- Support for custom oracle implementations

### 4. **Maintainability**
- Clear inheritance hierarchy
- Single responsibility principle
- Easy to test and debug individual components

## Running the Examples

### Test Individual Oracle Types
```bash
# Regex-based learning
python example.py

# RNN-based learning
python rnn_example.py
```

### Compare Both Approaches
```bash
# Side-by-side comparison
python comparison_example.py
```

## Expected Results

When running the comparison example, you should see:
- Both oracles successfully learn DFAs for the same language
- 100% agreement between the learned DFAs
- Similar state counts (typically 3 states for the "contains 01" pattern)
- Different query counts (RNN typically requires more queries due to training overhead)

## Implementation Notes

### Abstract Method Requirement
All concrete membership oracles must implement:
```python
def query(self, word: str) -> bool:
    """Return True iff word is in the target language."""
    pass
```

### Optional Extensions
Concrete oracles can add additional methods:
- `get_query_count()`: Track number of queries
- `get_confidence()`: Confidence scores for decisions
- `print_statistics()`: Detailed analysis output

### Integration with L* Framework
The abstract design ensures that:
- All oracles work with existing `EquivalenceOracle` implementations
- `LStarLearner` requires no changes to support new oracle types
- Observation table construction is oracle-agnostic

## Future Extensions

The abstract architecture makes it easy to add:
- File-based oracles (reading from datasets)
- Interactive oracles (human-in-the-loop learning)
- Composite oracles (combining multiple sources)
- Cached oracles (memoization for expensive queries)
- Remote oracles (API-based membership checking)

This design provides a solid foundation for experimenting with different approaches to membership oracle implementation while maintaining compatibility with the core L* algorithm. 