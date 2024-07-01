from nn import MLP

if __name__ == "__main__":
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    n = MLP(3, [4, 4, 1])
    ys = [1.0, -1.0, 1.0, 1.0]

    for round in range(100):
        ypred = [n(x) for x in xs]
        loss = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)])

        print(f"{round} round ({loss.data}) : {ypred}")
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        for p in n.parameters():
            p.data += -0.1 * p.grad
