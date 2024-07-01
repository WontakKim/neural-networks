import math


class Value:
    def __init__(self, data, children=(), operator="", label=""):
        self.data = data
        self.grad = 0.0
        self.previous = set(children)
        self.operator = operator
        self.label = label

        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data=self.data + other.data,
            children=(self, other),
            operator="+"
        )

        def _backward():
            """
            db/da = [{(a + h) + b} - {a + b}] / h
                = (a + b + h - a - b) / h
                = h / h
                = 1

            da/db = [{a + (b + h)} - {a + b}] / h
                = (a + b + h - a - b) / h
                = h / h
                = 1
            """
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data=self.data * other.data,
            children=(self, other),
            operator="*"
        )

        def _backward():
            """
            db/da = [{(a + h) * b} - {a * b}] / h
                = (ab + hb - ab) / h
                = hb / h
                = b

            da/db = [{a * (b + h)} - {a * b}] / h
                = (ab + ah - ab) / h
                = ah / h
                = a
            """
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data=self.data**other.data,
            children=(self,),
            operator=f"**{other.data}"
        )

        """
        d(x^n)/dx = nx^(n-1)
        db/da = [{(a + h)^b} - a^b] / h
            = [{a^b + ba^(b-1)h + bC2a^(b-2)h + ... + h^b} - a^b] / h
            = [ba^(b-1)h + bC2a^(b-2)h ... + h^b] / h
            = ba^(b-1)
        """
        def _backward():
            self.grad += (other.data * self.data ** (other.data - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        tanh = (math.exp(self.data * 2) - 1) / (math.exp(self.data * 2) + 1)
        out = Value(
            data=tanh,
            children=(self,),
            operator="tanh"
        )

        """
        tanh(x) = sinh(x) / cosh(x) = (e^(2*x) - 1) / (e^(2*x) + 1)
        d/dx * tanh(x) = 1 - tanh(x)^2
        """
        def _backward():
            self.grad += (1 - tanh**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = self._build_topological_list()
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def _build_topological_list(self):
        out = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v.previous:
                    build(child)
                out.append(v)
        build(self)
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return self * other**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
