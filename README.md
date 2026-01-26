# Neural Network Autocomplete from Scratch

Character-level language model and autocomplete engine built entirely from first principles using only NumPy. Inspired by Andrej Karpathy's "Neural Networks: Zero to Hero" course.

## Why I Built This

Pretty simple:
- Thought it was super cool to see how neural nets actually work under the hood
- Learned a ton about gradient computation and backprop that PyTorch hides from you
- It can suggest new words based on your input, which is neat
- Building everything from scratch really makes you understand what's going on

## Overview

This project implements a complete neural network framework from scratch, including custom autograd engine, backpropagation, and optimization algorithms. The model predicts the next character in a sequence and provides real-time autocomplete suggestions.

## Architecture

- Custom Value class with automatic differentiation
- Feedforward neural network with configurable hidden layers
- Batch normalization for training stability
- Adam optimizer with momentum and adaptive learning rates
- Context window of configurable size (default: 4 characters)
- Cross-entropy loss for character prediction

## Project Structure

```
autopredict/
├── src/
│   └── autopredict/
│       ├── value.py      - Autograd engine with gradient computation
│       ├── modules.py    - Neural network layers (Linear, BatchNorm, Model)
│       ├── loss.py       - CrossEntropyLoss implementation
│       ├── data.py       - Data loading and preprocessing
│       └── __init__.py   - Package initialization
├── scripts/
│   ├── train.py          - Training script with CLI arguments
│   ├── generate.py       - Text generation and prediction
│   └── eval.py           - Evaluation metrics (accuracy, perplexity, top-k)
├── terminal/
│   └── main.py           - Interactive autocomplete interface
├── data/                 - Training datasets
├── weights/              - Saved model weights and config
├── Dockerfile            - Docker container configuration
├── requirements.txt      - Python dependencies
├── setup.py              - Package setup configuration
└── .dockerignore         
```

## Installation

### Local Installation

```bash
pip install -r requirements.txt
```

Or install packages directly:

```bash
pip install numpy torch keyboard
```

### Docker Installation

Build the Docker image:

```bash
docker build -t autopredict .
```

## Usage

### Using Docker

Train the model:

```bash
docker run -v $(pwd)/weights:/app/weights -v $(pwd)/data:/app/data autopredict python scripts/train.py --training-set data/names.txt --epochs 50
```

Generate text:

```bash
docker run -v $(pwd)/weights:/app/weights autopredict python scripts/generate.py --num_samples 20
```

### Local Usage

### Training

Train the model on your dataset:

```bash
python scripts/train.py \
  --training-set data/names.txt \
  --lr 0.003 \
  --epochs 50 \
  --batch_size 128 \
  --emb_dim 16 \
  --hidden_size 128 \
  --context_size 4 \
  --weight_decay 0.001
```

### Generation

Generate text samples:

```bash
python scripts/generate.py --num_samples 20 --temperature 1.5
```

### Interactive Autocomplete

Real-time autocomplete in terminal (local use only):

```bash
cd terminal
python main.py
```

Controls:
- Type normally to see suggestions
- TAB to accept suggestion
- SPACE to accept and add space
- ESC to exit

Note: Interactive mode requires local installation due to keyboard library dependencies.

## Training Parameters

- `--training-set`: Path to training data file
- `--lr`: Learning rate (default: 0.01)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Training batch size (default: 32)
- `--emb_dim`: Character embedding dimension (default: 16)
- `--hidden_size`: Hidden layer size (default: 128)
- `--context_size`: Context window size (default: 4)
- `--weight_decay`: L2 regularization strength (default: 0.0001)
- `--train-frac`: Training set fraction (default: 0.8)
- `--dev-frac`: Dev set fraction (default: 0.1)

## Implementation Details

### Custom Autograd Engine

Implements automatic differentiation with:
- Forward pass computation
- Backward pass gradient accumulation
- Broadcasting support for operations
- Topological sorting for backpropagation

### Optimization Techniques

- Adam optimizer with bias correction
- Batch normalization with running statistics
- L2 weight regularization
- Learning rate decay schedule

### Data Processing

- Train/dev/test split with reproducible randomization
- Character-level tokenization
- Sliding context window for n-gram modeling
- Efficient batch iteration

## Model Performance

The model learns to predict characters based on context. Performance depends on:
- Dataset size and quality
- Context window size
- Model capacity (hidden size, embedding dim)
- Regularization strength

Example test loss: ~1.9 on 10k English words (random baseline: ~3.3)

## Why I Built This

Pretty simple:
- Thought it was super cool to see how neural nets actually work under the hood
- Learned a ton about gradient computation and backprop that PyTorch hides from you
- It can suggest new words based on your input, which is neat
- Building everything from scratch really makes you understand what's going on

## Acknowledgments

Inspired by Andrej Karpathy's course on building neural networks from scratch.
