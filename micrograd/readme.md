## Micrograd

The repository for learning Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd) with backpropagation

### Derivative([link](https://en.wikipedia.org/wiki/Derivative))

```math
L = \lim\limits_{h \to \infty} \frac{f(a + h)-f(a)}{h}
```

### Chain rule ([link](https://en.wikipedia.org/wiki/Chain_rule))

```math
\frac{dz}{dx} = \frac{dz}{dy} * \frac{dy}{dx}
```

### Hyperbolic functions ([link](https://en.wikipedia.org/wiki/Hyperbolic_functions))

- Hyperbolic tangent:

```math
\tanh{x} = \frac{\sinh{x}}{\cosh{x}} = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}
```

- Derivative tangent:
```math
\frac{d}{dx} \tanh{x} = 1 - \tanh^2{x}
```
