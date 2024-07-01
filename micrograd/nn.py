import random

from value import Value
from visualizer import draw_dot


class Neuron:
    """
    link : https://cs231n.github.io/neural-networks-1/
    """
    def __init__(self, input_size):
        self.w = [Value(random.uniform(-1, 1), label="w") for _ in range(input_size)]
        self.b = Value(random.uniform(-1, 1), label="b")

    def __call__(self, arguments):
        activation = sum((wi * xi for wi, xi in zip(self.w, arguments)), self.b)
        return activation.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, input_size, neuron_count):
        self.neurons = [Neuron(input_size) for _ in range(neuron_count)]

    def __call__(self, arguments):
        outputs = [neuron(arguments) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [params for neuron in self.neurons for params in neuron.parameters()]


class MLP:
    def __init__(self, input_size, output_sizes):
        sizes = [input_size] + output_sizes
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(output_sizes))]

    def __call__(self, arguments):
        for layer in self.layers:
            arguments = layer(arguments)
        return arguments

    def parameters(self):
        return [params for layer in self.layers for params in layer.parameters()]

