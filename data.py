import numpy as np

def load_vocab(filepath='names.txt'):

    words = open(filepath, 'r').read().splitlines()

    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    return words, stoi, itos

def dataloader(words, stoi, batch_size=32):

    xs, ys = [], []

    for w in words:

        chs = ['.'] + list(w) + ['.']

        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):

            xs.append([stoi[ch1], stoi[ch2]])
            ys.append(stoi[ch3])

    xs = np.array(xs)
    ys = np.array(ys)

    for i in range(0, len(xs), batch_size):
        yield xs[i:i+batch_size], ys[i:i+batch_size]
