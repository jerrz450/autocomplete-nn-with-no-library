import numpy as np
import torch
import argparse
from value import Value
from modules import Model

def load_artifacts():

    config = np.load('weights/config.npy', allow_pickle=True).item()
    stoi = np.load('weights/stoi.npy', allow_pickle=True).item()
    itos = np.load('weights/itos.npy', allow_pickle=True).item()
    C = Value(np.load('weights/embeddings.npy'))

    return config, stoi, itos, C

def init_model(config, C, emb_dim=None, hidden_size=None):

    emb_dim = emb_dim or config['emb_dim']
    hidden_size = hidden_size or config['hidden_size']

    model = Model(
        emb_dim * config['context_size'],
        [hidden_size, config['vocab_size']]
    )

    for i, layer in enumerate(model.layers):

        layer.W.data = np.load(f'weights/layer{i}_W.npy')
        layer.b.data = np.load(f'weights/layer{i}_b.npy')

    for i, bn in enumerate(model.bns):

        bn.gamma.data = np.load(f'weights/bn{i}_gamma.npy')
        bn.beta.data = np.load(f'weights/bn{i}_beta.npy')
        bn.running_mean = np.load(f'weights/bn{i}_running_mean.npy')
        bn.running_var = np.load(f'weights/bn{i}_running_var.npy')
        bn.training = False

    return model

def generate(model, C, itos, config, num_samples=10, temperature=1.0):

    context_size = config['context_size']

    for _ in range(num_samples):

        out = []
        context = [0]

        while True:

            ctx = ([0] * (context_size - len(context)) + context)[-context_size:]
            
            xemb = Value(C.data[ctx].reshape(1, -1))
            logits = model(xemb)

            logits_temp = Value(logits.data / temperature)
            probs = logits_temp.softmax(dim=1).data[0]
            ix = torch.multinomial(torch.tensor(probs), 1).item()

            if ix == 0:
                break

            out.append(itos[ix])
            context.append(ix)

        print(''.join(out))

def predict(model, C, itos, stoi, config, temperature=1.0):

    context_size = config['context_size']
    full_context = []

    while True:

        command = yield

        if command is None:
            continue

        if isinstance(command, tuple):
            action, value = command

            if action == 'reset':
                full_context = []
                yield ""
                continue

            elif action == 'set_text':
                full_context = [stoi.get(c, 0) for c in value if c in stoi]
                yield ""
                continue

        if command not in stoi:
            yield ""
            continue

        char_lookup = stoi[command]
        full_context.append(char_lookup)

        suggestion = []
        temp_context = full_context[:]

        for _ in range(20):
            ctx = ([0] * (context_size - len(temp_context)) + temp_context)[-context_size:]

            xemb = Value(C.data[ctx].reshape(1, -1))
            logits = model(xemb)

            logits_temp = Value(logits.data / temperature)
            probs = logits_temp.softmax(dim=1).data[0]
            ix = torch.multinomial(torch.tensor(probs), 1).item()

            if ix == 0:
                break

            suggestion.append(itos[ix])
            temp_context.append(ix)

        yield ''.join(suggestion)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--emb_dim', type=int, default=None)
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    config, stoi, itos, C = load_artifacts()
    model = init_model(config, C, args.emb_dim, args.hidden_size)

    generate(
        model,
        C,
        itos,
        config,
        num_samples=args.num_samples,
        temperature=args.temperature
    )