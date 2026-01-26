import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def accuracy(predictions, targets):
    return (predictions == targets).sum() / len(targets)

def perplexity(loss):
    return np.exp(loss)

def top_k_accuracy(logits, targets, k=3):
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]
    correct = np.array([targets[i] in top_k_preds[i] for i in range(len(targets))])
    return correct.sum() / len(targets)

def plot_eval_metrics(epochs, train_losses, dev_losses, train_accs, dev_accs, output_path='weights/training_metrics.png'):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs_range = range(1, epochs + 1)

    axes[0].plot(epochs_range, train_losses, label='Train')
    axes[0].plot(epochs_range, dev_losses, label='Dev')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, [acc * 100 for acc in train_accs], label='Train')
    axes[1].plot(epochs_range, [acc * 100 for acc in dev_accs], label='Dev')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].legend()
    axes[1].grid(True)

    train_perplexities = [perplexity(loss) for loss in train_losses]
    dev_perplexities = [perplexity(loss) for loss in dev_losses]
    axes[2].plot(epochs_range, train_perplexities, label='Train')
    axes[2].plot(epochs_range, dev_perplexities, label='Dev')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Perplexity')
    axes[2].set_title('Perplexity over Epochs')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Training plots saved to {output_path}")
