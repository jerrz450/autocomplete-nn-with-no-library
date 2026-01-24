import numpy as np
from value import Value
from modules import Model
from loss import CrossEntropyLoss
from data import load_vocab, dataloader

def train(
    lr = 0.01,
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,
    weight_decay = 0.0001,
    epochs = 20
    ):

    words, stoi, itos = load_vocab('names.txt')

    C = Value(np.random.randn(27, 16) * 0.01)
    model = Model(32, [128, 27])
    loss_fn = CrossEntropyLoss()

    for p in model.parameters():
        p.data *= 0.1

    params = model.parameters() + [C]
    m = [np.zeros_like(p.data) for p in params]
    v = [np.zeros_like(p.data) for p in params]

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0

        for xb, yb in dataloader(words, stoi, batch_size=32):
            xemb = Value(C.data[xb].reshape(len(xb), -1))

            logits = model(xemb)
            loss = loss_fn(logits, yb)

            reg_loss = sum((p.data ** 2).sum() for p in model.parameters()) * weight_decay
            total_loss_with_reg = loss.data + reg_loss

            loss.backward()

            for i, p in enumerate(params[:-1]):
                p.grad += 2 * weight_decay * p.data

            C.grad[xb.flatten()] += xemb.grad.reshape(-1, 16)

            for i, p in enumerate(params):
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                v[i] = beta2 * v[i] + (1 - beta2) * (p.grad ** 2)
                m_hat = m[i] / (1 - beta1 ** (count + 1))
                v_hat = v[i] / (1 - beta2 ** (count + 1))
                p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)
                p.grad[:] = 0

            total_loss += total_loss_with_reg
            count += 1

        print(f"Epoch {epoch}, Loss: {total_loss/count:.4f}")

        if epoch % 10 == 9:
            lr *= 0.5

    np.save('weights/embeddings.npy', C.data)
    for i, layer in enumerate(model.layers):
        np.save(f'weights/layer{i}_W.npy', layer.W.data)
        np.save(f'weights/layer{i}_b.npy', layer.b.data)

    for i, bn in enumerate(model.bns):
        np.save(f'weights/bn{i}_gamma.npy', bn.gamma.data)
        np.save(f'weights/bn{i}_beta.npy', bn.beta.data)
        np.save(f'weights/bn{i}_running_mean.npy', bn.running_mean)
        np.save(f'weights/bn{i}_running_var.npy', bn.running_var)

    print("Model saved to weights/")

if __name__ == "__main__":
    train()