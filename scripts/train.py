import numpy as np
import argparse
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WEIGHTS_DIR = PROJECT_ROOT / 'weights'
WEIGHTS_DIR.mkdir(exist_ok=True)

from src.autopredict.value import Value
from src.autopredict.modules import Model
from src.autopredict.loss import CrossEntropyLoss
from src.autopredict.data import load_vocab, split_data, build_dataset, dataloader
from eval import accuracy, perplexity, top_k_accuracy, plot_eval_metrics

def train_epoch(model, C, train_words, stoi, loss_fn, params, m, v, beta1, beta2, eps, lr, weight_decay, batch_size, context_size, emb_dim, step_count):

    total_loss = 0.0
    all_predictions = []
    all_targets = []
    count = 0

    for xb, yb in dataloader(train_words, stoi, batch_size=batch_size, context_size=context_size):

        xemb = Value(C.data[xb].reshape(len(xb), -1))
        logits = model(xemb)
        loss = loss_fn(logits, yb)

        predictions = logits.data.argmax(axis=1)
        all_predictions.extend(predictions)
        all_targets.extend(yb)

        reg_loss = sum((p.data ** 2).sum() for p in model.parameters()) * weight_decay
        total_loss_with_reg = loss.data + reg_loss

        loss.backward()

        for i, p in enumerate(params[:-1]):
            p.grad += 2 * weight_decay * p.data

        C.grad[xb.flatten()] += xemb.grad.reshape(-1, emb_dim)

        for i, p in enumerate(params):

            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * (p.grad ** 2)
            m_hat = m[i] / (1 - beta1 ** (step_count + count + 1))
            v_hat = v[i] / (1 - beta2 ** (step_count + count + 1))
            p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)
            p.grad[:] = 0

        total_loss += total_loss_with_reg
        count += 1

    avg_loss = total_loss / count
    avg_acc = accuracy(np.array(all_predictions), np.array(all_targets))
    return avg_loss, avg_acc, count

def evaluate(model, C, eval_words, stoi, loss_fn, batch_size, context_size):

    for bn in model.bns:
        bn.training = False

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for xb, yb in dataloader(eval_words, stoi, batch_size=batch_size, context_size=context_size):

        xemb = Value(C.data[xb].reshape(len(xb), -1))
        logits = model(xemb)
        loss = loss_fn(logits, yb)

        predictions = logits.data.argmax(axis=1)
        all_predictions.extend(predictions)
        all_targets.extend(yb)

        total_loss += loss.data * len(yb)

    for bn in model.bns:
        bn.training = True

    avg_loss = total_loss / len(all_targets)
    avg_acc = accuracy(np.array(all_predictions), np.array(all_targets))

    return avg_loss, avg_acc

def train(
    training_set='names.txt',
    lr=0.01,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0001,
    epochs=20,
    batch_size=32,
    emb_dim=16,
    hidden_size=128,
    context_size=4,
    train_frac=0.8,
    dev_frac=0.1
    ):

    words, stoi, itos = load_vocab(training_set)
    train_words, dev_words, test_words = split_data(words, train_frac, dev_frac)

    vocab_size = len(stoi)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Context size: {context_size}")
    print(f"Dataset splits: {len(train_words)} train, {len(dev_words)} dev, {len(test_words)} test")

    C = Value(np.random.randn(vocab_size, emb_dim) * 0.01)

    model = Model(emb_dim * context_size, [hidden_size, vocab_size])
    loss_fn = CrossEntropyLoss()

    for p in model.parameters():
        p.data *= 0.1

    params = model.parameters() + [C]
    m = [np.zeros_like(p.data) for p in params]
    v = [np.zeros_like(p.data) for p in params]

    train_losses = []
    dev_losses = []
    train_accs = []
    dev_accs = []
    step_count = 0

    for epoch in range(epochs):

        train_loss, train_acc, batch_count = train_epoch(
            model, C, train_words, stoi, loss_fn, params, m, v,
            beta1, beta2, eps, lr, weight_decay, batch_size, context_size, emb_dim, step_count
        )
        step_count += batch_count

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        dev_loss, dev_acc = evaluate(model, C, dev_words, stoi, loss_fn, batch_size, context_size)
        dev_losses.append(dev_loss)
        dev_accs.append(dev_acc)

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc*100:.2f}%")

        if epoch % 10 == 9:
            lr *= 0.5

    test_loss, test_acc = evaluate(model, C, test_words, stoi, loss_fn, batch_size, context_size)

    print(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")

    np.save(WEIGHTS_DIR / 'embeddings.npy', C.data)
    np.save(WEIGHTS_DIR / 'config.npy', {'emb_dim': emb_dim, 'hidden_size': hidden_size, 'vocab_size': vocab_size, 'context_size': context_size})
    np.save(WEIGHTS_DIR / 'stoi.npy', stoi)
    np.save(WEIGHTS_DIR / 'itos.npy', itos)

    for i, layer in enumerate(model.layers):
        np.save(WEIGHTS_DIR / f'layer{i}_W.npy', layer.W.data)
        np.save(WEIGHTS_DIR / f'layer{i}_b.npy', layer.b.data)

    for i, bn in enumerate(model.bns):
        np.save(WEIGHTS_DIR / f'bn{i}_gamma.npy', bn.gamma.data)
        np.save(WEIGHTS_DIR / f'bn{i}_beta.npy', bn.beta.data)
        np.save(WEIGHTS_DIR / f'bn{i}_running_mean.npy', bn.running_mean)
        np.save(WEIGHTS_DIR / f'bn{i}_running_var.npy', bn.running_var)

    print(f"Model saved to {WEIGHTS_DIR}")

    plot_eval_metrics(epochs, train_losses, dev_losses, train_accs, dev_accs, WEIGHTS_DIR / 'training_metrics.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-set', type=str, default='names.txt')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--emb_dim', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--context_size', type=int, default=4)
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--dev-frac', type=float, default=0.1)
    args = parser.parse_args()

    train(
        training_set=args.training_set,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_size,
        context_size=args.context_size,
        train_frac=args.train_frac,
        dev_frac=args.dev_frac
    )