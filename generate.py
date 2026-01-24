import numpy as np
import torch
import argparse
from value import Value
from modules import Model

def generate(num_samples=10, emb_dim=None, hidden_size=None, temperature=1.0):

    config = np.load('weights/config.npy', allow_pickle=True).item()
    stoi = np.load('weights/stoi.npy', allow_pickle=True).item()
    itos = np.load('weights/itos.npy', allow_pickle=True).item()

    if emb_dim is None:
        emb_dim = config['emb_dim']

    if hidden_size is None:
        hidden_size = config['hidden_size']

    vocab_size = config['vocab_size']
    context_size = config['context_size']

    C = Value(np.load('weights/embeddings.npy'))
    model = Model(emb_dim * context_size, [hidden_size, vocab_size])

    for i, layer in enumerate(model.layers):

        layer.W.data = np.load(f'weights/layer{i}_W.npy')
        layer.b.data = np.load(f'weights/layer{i}_b.npy')

    for i, bn in enumerate(model.bns):

        bn.gamma.data = np.load(f'weights/bn{i}_gamma.npy')
        bn.beta.data = np.load(f'weights/bn{i}_beta.npy')
        bn.running_mean = np.load(f'weights/bn{i}_running_mean.npy')
        bn.running_var = np.load(f'weights/bn{i}_running_var.npy')
        bn.training = False

    for _ in range(num_samples):

        out = []
        context = [0] * context_size

        while True:

            xemb = Value(C.data[context].reshape(1, -1))
            logits = model(xemb)

            logits_temp = Value(logits.data / temperature)
            probs = logits_temp.softmax(dim=1).data[0]
            probs = probs / probs.sum()

            ix = torch.multinomial(torch.tensor(probs), 1).item()

            out.append(itos[ix])
            context = context[1:] + [ix]

            if ix == 0 and len(out) >= 2:
                break

        print(''.join(out))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--emb_dim', type=int, default=None)
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    generate(
        num_samples=args.num_samples,
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_size,
        temperature=args.temperature
    )
