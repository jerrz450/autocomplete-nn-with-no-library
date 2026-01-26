import numpy as np
from .value import Value

class Linear:

    def __init__(self, n_inputs, n_outputs):

        self.W = Value(np.random.uniform(-1, 1, (n_outputs, n_inputs)))
        self.b = Value(np.random.uniform(-1, 1, (n_outputs,)))

    def __call__(self, x):

        out = Value(x.data @ self.W.data.T, (x, self.W))

        def _backward():
            x.grad += out.grad @ self.W.data
            self.W.grad += out.grad.T @ x.data

        out._backward = _backward
        return out + self.b

    def parameters(self):

        return [self.W, self.b]

class BatchNorm:

    def __init__(self, n_features, momentum=0.1, eps=1e-5):
        self.gamma = Value(np.ones(n_features))
        self.beta = Value(np.zeros(n_features))
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        self.training = True

    def __call__(self, x):
        if self.training:
            mean = x.data.mean(axis=0)
            var = x.data.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        std = np.sqrt(var + self.eps)
        xnorm = Value((x.data - mean) / std, (x,))

        def _backward():
            if self.training:
                dxnorm = xnorm.grad
                N = x.data.shape[0]
                x.grad += (dxnorm - dxnorm.mean(axis=0) - xnorm.data * (dxnorm * xnorm.data).mean(axis=0)) / std
            else:
                x.grad += xnorm.grad / std

        xnorm._backward = _backward
        out = xnorm * self.gamma + self.beta
        return out

    def parameters(self):
        return [self.gamma, self.beta]

class Model:

    def __init__(self, nin, nouts):

        sizes = [nin] + nouts
        self.layers = [Linear(sizes[i], sizes[i + 1]) for i in range(len(nouts))]
        self.bns = [BatchNorm(sizes[i + 1]) for i in range(len(nouts) - 1)]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.bns[i](x)
                x = x.relu()
        return x


    def parameters(self):

        params = [p for layer in self.layers for p in layer.parameters()]
        params += [p for bn in self.bns for p in bn.parameters()]
        return params
