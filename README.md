# L* CLI: Extract DFAs from Any RNN

**A powerful command-line tool that uses the L* active learning algorithm to extract interpretable Deterministic Finite Automata (DFA) from any type of Recurrent Neural Network.**

Transform your black-box RNNs into understandable, formal models with a single command.

## 🎯 What This Tool Does

The L* CLI tool takes **any** trained RNN and automatically learns an equivalent DFA that captures the language patterns the RNN has learned. This enables:

- **Neural Network Interpretability**: Understand what your RNN actually learned
- **Formal Verification**: Convert neural networks to verifiable automata
- **Pattern Discovery**: Extract explicit rules from implicit neural representations
- **Model Debugging**: Visualize decision boundaries as state machines

## 🚀 Quick Start

### Installation
```bash
git clone <repository-url>
cd Lstar
pip install -r requirements.txt
```

### Basic Usage
```bash
# Extract DFA from any PyTorch model
python lstar_cli/lstar_extract.py \
  --model-type pytorch \
  --model-path your_model.pth \
  --model-class YourRNNClass \
  --alphabet "01" \
  --output learned_dfa.png

# Extract DFA from a custom Python function
python lstar_cli/lstar_extract.py \
  --model-type function \
  --model-path classifier.py \
  --function-name my_classifier \
  --alphabet "abc" \
  --output dfa.png

# Extract DFA from any Hugging Face model
python lstar_cli/lstar_extract.py \
  --model-type huggingface \
  --model-name bert-base-uncased \
  --alphabet "abcdefghijklmnopqrstuvwxyz " \
  --output sentiment_automaton.png
```

## 🔧 Supported RNN Types

The CLI tool works with **any** RNN implementation through flexible adapters:

### 1. PyTorch Models
- **LSTM, GRU, RNN**: Any PyTorch recurrent architecture
- **Custom Models**: Your own neural network implementations
- **Pretrained Models**: Load from `.pth` files with custom parameters

```bash
python lstar_cli/lstar_extract.py \
  --model-type pytorch \
  --model-path model.pth \
  --model-class MyLSTM \
  --model-kwargs '{"hidden_size": 128, "num_layers": 2}' \
  --alphabet "01" \
  --output dfa.png
```

### 2. Custom Functions
- **Any Python Function**: Turn any classification function into an "RNN"
- **Legacy Code**: Extract patterns from existing rule-based systems
- **Hybrid Models**: Combine neural and symbolic approaches

```bash
python lstar_cli/lstar_extract.py \
  --model-type function \
  --model-path my_classifier.py \
  --function-name predict_sentiment \
  --alphabet "abcdefghijklmnopqrstuvwxyz .,!?" \
  --output sentiment_dfa.png
```

### 3. Hugging Face Models
- **Transformers**: BERT, GPT, RoBERTa, and more
- **Sequence Classification**: Any HF model for text classification
- **Custom Tokenization**: Automatic handling of model vocabularies

```bash
python lstar_cli/lstar_extract.py \
  --model-type huggingface \
  --model-name distilbert-base-uncased \
  --alphabet "hello world the quick brown fox" \
  --output transformer_dfa.png
```

## ⚙️ CLI Options

### Core Arguments
- `--model-type`: `pytorch`, `function`, or `huggingface`
- `--alphabet`: Input alphabet as string (`"abc"`) or JSON list (`["a","b","c"]`)
- `--output`: Output file (`.png` for visualization, `.dot` for graphviz)

### PyTorch Specific
- `--model-path`: Path to `.pth` model file
- `--model-class`: Python class name of your model
- `--model-module`: Python file containing the model class
- `--model-kwargs`: JSON string of model constructor arguments
- `--device`: `cpu` or `cuda` for GPU acceleration

### Function Specific
- `--model-path`: Path to `.py` file containing your function
- `--function-name`: Name of the prediction function

### Hugging Face Specific
- `--model-name`: HuggingFace model identifier
- `--model-path`: Local path to HuggingFace model

### L* Algorithm Tuning
- `--threshold`: Classification threshold (default: 0.5)
- `--max-prefix-len`: Maximum prefix length for exploration (default: 3)
- `--max-suffix-len`: Maximum suffix length for testing (default: 4)
- `--verbose`: Enable detailed learning statistics

### Testing & Debugging
- `--test-strings`: Comma-separated test inputs to evaluate
- `--verbose`: Show query statistics and learning progress

## 📊 Example Workflows

### 1. Sentiment Analysis Model
```bash
# Extract DFA from a sentiment classifier
python lstar_cli/lstar_extract.py \
  --model-type pytorch \
  --model-path sentiment_lstm.pth \
  --model-class SentimentLSTM \
  --alphabet "abcdefghijklmnopqrstuvwxyz .,!?" \
  --test-strings "good,bad,excellent,terrible" \
  --verbose \
  --output sentiment_rules.png
```

### 2. Sequence Pattern Recognition
```bash
# Learn patterns from a sequence classifier
python lstar_cli/lstar_extract.py \
  --model-type function \
  --model-path pattern_detector.py \
  --function-name detect_pattern \
  --alphabet "01" \
  --test-strings "01,10,0011,1100,010101" \
  --output pattern_automaton.png
```

### 3. Language Model Analysis
```bash
# Extract decision patterns from a transformer
python lstar_cli/lstar_extract.py \
  --model-type huggingface \
  --model-name gpt2 \
  --alphabet "the and or but" \
  --max-prefix-len 4 \
  --max-suffix-len 5 \
  --verbose \
  --output language_patterns.png
```

## 🔬 Understanding the Output

The tool generates:

1. **DFA Visualization** (`.png`): State machine diagram showing:
   - States as circles (accepting states are double-circled)
   - Transitions labeled with alphabet symbols
   - Clear visual representation of learned patterns

2. **Graphviz Source** (`.dot`): Text representation for custom styling

3. **Learning Statistics** (with `--verbose`):
   - Number of queries made to the RNN
   - Learning time and efficiency metrics
   - Confidence distributions and pattern analysis

## 🏗️ Architecture

```
Lstar/
├── lstar_cli/           # 🎯 Main CLI tool
│   └── lstar_extract.py # Universal RNN→DFA extractor
├── lstar/               # Core L* algorithm
├── rnn_adapters/        # Universal RNN interfaces
├── examples/            # Usage examples
└── docs/               # Documentation
```

## 🔧 Advanced Usage

### Custom Model Integration
Create your own model adapter:

```python
# my_custom_model.py
def my_classifier(text):
    # Your classification logic here
    return confidence_score  # 0.0 to 1.0

# Use with CLI:
python lstar_cli/lstar_extract.py \
  --model-type function \
  --model-path my_custom_model.py \
  --function-name my_classifier \
  --alphabet "your_alphabet" \
  --output custom_dfa.png
```

### Batch Processing
Process multiple models:

```bash
# Extract DFAs from multiple models
for model in model1.pth model2.pth model3.pth; do
  python lstar_cli/lstar_extract.py \
    --model-type pytorch \
    --model-path $model \
    --model-class MyRNN \
    --alphabet "01" \
    --output ${model%.pth}_dfa.png
done
```

### Performance Tuning
Optimize for your use case:

```bash
# Fast extraction (fewer queries)
python lstar_cli/lstar_extract.py \
  --max-prefix-len 2 \
  --max-suffix-len 3 \
  [other options...]

# Thorough extraction (more accurate)
python lstar_cli/lstar_extract.py \
  --max-prefix-len 5 \
  --max-suffix-len 6 \
  [other options...]
```

## 📚 Documentation

- **[CLI Guide](docs/CLI_README.md)**: Complete command-line reference
- **[Integration Guide](docs/USER_GUIDE.md)**: Adding custom RNN types
- **[Architecture](docs/README_Architecture.md)**: Technical implementation details

## 🎓 Research Applications

This tool enables research in:
- **Neural Network Interpretability**: Understanding learned representations
- **Formal Verification**: Converting NNs to verifiable models
- **Model Compression**: Extracting minimal equivalent automata
- **Educational**: Teaching automata theory through neural networks

## 🤝 Contributing

Add support for new RNN frameworks:
1. Implement adapter in `rnn_adapters/`
2. Add CLI integration in `lstar_cli/lstar_extract.py`
3. Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **CLI Help**: `python lstar_cli/lstar_extract.py --help`
- **Examples**: See `examples/usage/` for working demos
- **Issues**: Report bugs and feature requests via GitHub issues

---

**Transform any RNN into an interpretable automaton with a single command!** 