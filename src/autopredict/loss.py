import numpy as np
from .value import Value

class CrossEntropyLoss:
    
    def __call__(self, logits, targets):

        shifted = logits - logits.data.max(axis=-1, keepdims=True)
        exp = shifted.exp()
        probs = exp / exp.sum(dim=-1, keepdim=True)

        p = probs.data[range(len(targets)), targets]
        loss = Value(-np.log(p).mean())

        def _backward():
            grad = probs.data
            grad[range(len(targets)), targets] -= 1
            grad /= len(targets)
            logits.grad += grad

        loss._backward = _backward
        loss._prev = {logits}
        return loss
