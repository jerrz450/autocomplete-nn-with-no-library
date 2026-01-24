# autocomplete-nn-with-no-library

Autocomplete model from scratch inspired by Andrej Karpathy's course.

I implemented all pytorch functionality for this simple neural net from scratch with only numpy, similar that pytorch does it (running on CPU of course).

# Neural Network Autocomplete from Scratch

Character-level autocomplete engine built from first principles, inspired by Andrej Karpathy's "Neural Networks: Zero to Hero" course.

## What It covers

- Custom autograd engine with automatic differentiation
- Manual backpropagation through computation graphs
- Gradient descent optimization from scratch
- Trigram language model architecture
- Optimization techniques (Batch Norm and Adam Grad)

## Features

- **Pure NumPy implementation** - Understanding neural nets at the lowest level
- **Real-time predictions** - Top-k character suggestions with probabilities
- **Training** - Can be trained on any small dataset of words
- **CPU-optimized** - Efficient inference without GPU dependencies

## Quick Start
```python
# Training
python train.py --epochs 100 --lr 0.1

# Inference
python predict.py --context "hel"
# Output: l (0.42), p (0.31), m (0.18)
```

## Takeaway from this

Building this taught me how modern ML frameworks work under the hood - gradient computation, backpropagation mechanics, and optimization algorithms that PyTorch abstracts away.

## Acknowledgments

Inspired by Andrej Karpathy's educational content on building neural networks from scratch.
