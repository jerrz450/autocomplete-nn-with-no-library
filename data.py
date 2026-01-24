import numpy as np

def load_vocab(filepath='names.txt'):

    words = open(filepath, 'r').read().splitlines()

    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    return words, stoi, itos

def split_data(words, train_frac=0.8, dev_frac=0.1):

    n = len(words)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)

    rng = np.random.RandomState(42)
    indices = rng.permutation(n)

    train_words = [words[i] for i in indices[:n_train]]
    dev_words = [words[i] for i in indices[n_train:n_train+n_dev]]
    test_words = [words[i] for i in indices[n_train+n_dev:]]

    return train_words, dev_words, test_words

def build_dataset(words, stoi):

    xs, ys = [], []

    for w in words:

        chs = ['.'] + list(w) + ['.']

        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):

            xs.append([stoi[ch1], stoi[ch2]])
            ys.append(stoi[ch3])

    return np.array(xs), np.array(ys)

def dataloader(words, stoi, batch_size=32):

    xs, ys = build_dataset(words, stoi)

    for i in range(0, len(xs), batch_size):
        yield xs[i:i+batch_size], ys[i:i+batch_size]
