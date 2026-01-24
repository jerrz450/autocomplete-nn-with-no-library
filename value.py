import numpy as np

class Value:
    def __init__(self, data, _children=(), _op=''):

        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            if self.grad.shape != out.grad.shape:
                axis = tuple(range(len(out.grad.shape) - len(self.grad.shape)))
                self.grad += out.grad.sum(axis=axis)
            else:
                self.grad += out.grad

            if other.grad.shape != out.grad.shape:
                axis = tuple(range(len(out.grad.shape) - len(other.grad.shape)))
                other.grad += out.grad.sum(axis=axis)
            else:
                other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad

            if self.grad.shape != grad_self.shape:
                axis = tuple(range(len(grad_self.shape) - len(self.grad.shape)))
                self.grad += grad_self.sum(axis=axis)
            else:
                self.grad += grad_self

            if other.grad.shape != grad_other.shape:
                axis = tuple(range(len(grad_other.shape) - len(other.grad.shape)))
                other.grad += grad_other.sum(axis=axis)
            else:
                other.grad += grad_other

        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, power):
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward

        return out

    def sum(self, dim=None, keepdim=False):

        out = Value(self.data.sum(axis=dim, keepdims=keepdim), (self,), 'sum')

        def _backward():

            grad = out.grad
            if not keepdim and dim is not None:
                grad = np.expand_dims(grad, axis=dim)
            self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out

    def softmax(self, dim=-1):
        max_val = self.data.max(axis=dim, keepdims=True)
        shifted = Value(self.data - max_val)

        exp_vals = shifted.exp()
        sum_exp = exp_vals.sum(dim=dim, keepdim=True)

        out = Value(exp_vals.data / sum_exp.data)
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        t = np.exp(self.data)
        out = Value(t, (self,), 'exp')

        def _backward():
            self.grad += t * out.grad
        out._backward = _backward

        return out

    def __matmul__(self, other):

        prod = Value(self.data * other.data, _children= (self, other), _op = '@')
        out = prod.sum()

        def _backward():

            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()
