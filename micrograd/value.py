import math

from visualizer import draw_dot


class Value:
    """
    Derivative:

        L = (f(a + h) - f(a)) / h

    Chain rule:

        dz/dx = dz/dy * dy/dx

    """
    def __init__(self, data, children=(), operator="", label=""):
        self.data = data
        self.grad = 0.0
        self.previous = set(children)
        self.operator = operator
        self.label = label

        self._backward = lambda: None

    """
    f(a) = a + b
    """
    def __add__(self, other):
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
            self.grad = out.grad
            other.grad = out.grad

        out._backward = _backward
        return out

    """
    f(a) = a * b
    """
    def __mul__(self, other):
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
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _backward
        return out

    """
    f(a) = a^b
    """
    def __pow__(self, other):
        out = Value(
            data=self.data ** other,
            children=(self,),
            operator=f"**{other}"
        )

        """
        d(x^n)/dx = nx^(n-1)
        db/da = [{(a + h)^b} - a^b] / h
            = [{a^b + ba^(b-1)h + bC2a^(b-2)h ... + h^b} - a^b] / h
            = [ba^(b-1)h + bC2a^(b-2)h ... + h^b] / h
            = ba^(b-1)
        """
        def _backward():
            self.grad = (other.data * self.data ** (other.data - 1)) * out.grad

        out._backward = _backward
        return out

    """
    Ref : https://en.wikipedia.org/wiki/Hyperbolic_functions
    
    tanh(x) = sinh(x) / cosh(x) = (e^(2*x) - 1) / (e^(2*x) + 1)
    d/dx * tanh(x) = 1 - tanh(x)^2
    """
    def tanh(self):
        tanh = (math.exp(self.data * 2) - 1) / (math.exp(self.data * 2) + 1)
        out = Value(
            data=tanh,
            children=(self,),
            operator="tanh"
        )

        def _backward():
            self.grad = (1 - tanh ** 2) * out.grad

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

    def __repr__(self):
        return f"Value(data={self.data})"


def lol():
    h = 0.0001

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b
    e.label = 'e'
    d = e + c
    d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f
    L.label = 'L'
    L1 = L.data

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b
    e.label = 'e'
    d = e + c
    d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f
    L.label = 'L'
    L2 = L.data + h

    tanh = L.tanh()

    tanh.backward()

    dot = draw_dot(tanh)
    dot.render("./output/draw.gv", view=True)



if __name__ == "__main__":
    lol()
    # dot = draw_dot(L)
    # dot.render("./output/draw.gv", view=True)
