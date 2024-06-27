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
            self.grad = (other * self.data ** (other - 1)) * out.grad

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
    # inputs x1,x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')



    # weights w1,w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # bias of the neuron
    b = Value(6.8813735870195432, label='b')

    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = 'x1*w1'

    x2w2 = x2 * w2
    x2w2.label = 'x2*w2'

    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = 'x1*w1 + x2*w2'

    n = x1w1x2w2 + b
    n.label = 'n'
    o = n.tanh()
    o.label = 'o'

    pow = o**2
    pow.label = "pow"

    pow.backward()

    dot = draw_dot(pow)
    dot.render("./output/draw.gv", view=True)



if __name__ == "__main__":
    lol()
    # dot = draw_dot(L)
    # dot.render("./output/draw.gv", view=True)
