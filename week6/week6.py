import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.utilities.misc import os


def gradient_descent(
    loss_fn, grad_fn, X, y, alpha, max_updates, init_w, batch_size=None
):
    rng = np.random.default_rng(67)
    batch_size = batch_size if batch_size is not None else len(X)
    w = init_w
    steps = [w]
    losses = [loss_fn(X, w, y).item()]
    number_updates = 0
    while number_updates < max_updates:
        indices = rng.permutation(len(X))
        X_shuff = X[indices]
        y_shuff = y[indices]

        for batch in range(0, len(X), batch_size):
            X_batch = X_shuff[batch : batch + batch_size]
            y_batch = y_shuff[batch : batch + batch_size]

            w = w - alpha * grad_fn(X_batch, w, y_batch)
            number_updates += 1

            losses.append(loss_fn(X, w, y).item())
            steps.append(w)

    return steps, losses


def question_1a():

    plt.rcParams.update(
        {
            'figure.figsize': (12, 6),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.frameon': True,
            'lines.linewidth': 1,
            'lines.markersize': 3,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        }
    )
    rng = np.random.default_rng(67)

    m = 1000
    X = rng.normal(0.0, 1.0, size=(m, 2))
    noise = rng.normal(0.0, 1.0, size=(m, 1))
    W_star = np.array([3, 4])

    linear_regression = lambda X, w: (X @ w).reshape(X.shape[0], -1)
    y = linear_regression(X, W_star) + noise

    def loss(X, w, y):
        y_hat = linear_regression(X, w)
        errors = y_hat - y
        return np.sum(errors**2, axis=0) * (1 / (2 * len(X)))

    def dloss(X, w, y):
        y_hat = linear_regression(X, w)
        errors = y_hat - y
        return (1 / len(X)) * X.T @ errors

    space = np.linspace(0.5, 5, 100)
    W1, W2 = np.meshgrid(space, space)

    stacked = np.stack([W1.flatten(), W2.flatten()], axis=0)

    epochs = 80
    alpha = 0.5
    init_w = np.array([[1], [1]])

    steps, losses = gradient_descent(loss, dloss, X, y, alpha, epochs, init_w)

    _, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    ax1.set_title('Full GD Trajectory')
    ax1.set_xlabel('w1')
    ax1.set_ylabel('w2')

    ax1.contour(W1, W2, loss(X, stacked, y).reshape(W1.shape), levels=20, cmap='plasma')
    ax1.plot(*zip(*steps), color='blue', marker='o', label='Trajectory')
    ax1.scatter(*steps[-1], c='blue', label='Last Step')
    ax1.scatter(*W_star, c='red', s=30, label='Optimum')
    ax1.legend()

    ax2.set_title('Full GD Loss vs. Iterations')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Loss (log)')
    ax2.plot(range(epochs + 1), losses, color='blue', label='Loss')
    ax2.set_yscale('log')
    ax2.legend()

    plt.savefig('./images/question1a.png')
    plt.show()


def question_1b():
    plt.rcParams.update(
        {
            'figure.figsize': (12, 6),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.frameon': True,
            'lines.linewidth': 1,
            'lines.markersize': 3,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        }
    )
    rng = np.random.default_rng(67)

    m = 1000
    X = rng.normal(0.0, 1.0, size=(m, 2))
    eps = rng.normal(0.0, 1.0, size=(m, 1))
    W_star = np.array([3, 4])

    linear_regression = lambda X, w: (X @ w).reshape(X.shape[0], -1)
    y = linear_regression(X, W_star) + eps

    def loss(X, w, y):
        y_hat = linear_regression(X, w)
        erros = y_hat - y
        return np.sum(erros**2, axis=0) * (1 / (2 * len(X)))

    def dloss(X, w, y):
        y_hat = linear_regression(X, w)
        errors = y_hat - y
        return (1 / len(X)) * X.T @ errors

    space = np.linspace(0.5, 5, 100)
    W1, W2 = np.meshgrid(space, space)

    stacked = np.stack([W1.flatten(), W2.flatten()], axis=0)

    batch_sizes = [len(X), 5, 20]
    colors = ['blue', 'red', 'green']
    updates = 400
    alpha = 0.5

    all_steps = []
    all_losses = []
    for batch_size in batch_sizes:
        init_w = np.array([[1], [1]])
        steps, batch_losses = gradient_descent(
            loss, dloss, X, y, alpha, updates, init_w, batch_size
        )
        all_steps.append(steps)
        all_losses.append(batch_losses)

    fig_contour, ax_contour = plt.subplots()
    ax_contour.set_title('SGD Trajectories by Batch Size')
    ax_contour.set_xlabel('w1')
    ax_contour.set_ylabel('w2')
    ax_contour.contour(
        W1, W2, loss(X, stacked, y).reshape(W1.shape), levels=20, cmap='plasma'
    )
    for steps, batch_size, color in zip(all_steps, batch_sizes, colors):
        ax_contour.plot(
            *zip(*steps[::10]),
            color=color,
            marker='o',
            label=f'Batch size {batch_size}',
        )
        ax_contour.scatter(*steps[-1], c=color)
    ax_contour.scatter(*W_star, c='red', s=30, label='Optimum')
    ax_contour.legend()
    fig_contour.savefig('./images/question1b_trajectories.png')
    plt.show()

    for batch_losses, batch_size in zip(all_losses, batch_sizes):
        fig_loss, ax_loss = plt.subplots()
        ax_loss.set_title(f'SGD Loss vs. Updates (Batch size {batch_size})')
        ax_loss.set_xlabel('Updates')
        ax_loss.set_ylabel('Loss (log)')
        ax_loss.plot(
            range(updates + 1),
            batch_losses,
            color='blue',
            label=f'Batch size {batch_size}',
        )
        ax_loss.set_yscale('symlog')
        ax_loss.legend()
        fig_loss.savefig(f'./images/question1b_loss_batch_size-{batch_size}.png')
        plt.show()


def question_1c():
    plt.rcParams.update(
        {
            'figure.figsize': (12, 6),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.frameon': True,
            'lines.linewidth': 1,
            'lines.markersize': 3,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        }
    )
    rng = np.random.default_rng(67)

    m = 1000
    X = rng.normal(0.0, 1.0, size=(m, 2))
    eps = rng.normal(0.0, 1.0, size=(m, 1))
    W_star = np.array([3, 4])

    linear_regression = lambda X, w: (X @ w).reshape(X.shape[0], -1)
    y = linear_regression(X, W_star) + eps

    def loss(X, w, y):
        y_hat = linear_regression(X, w)
        erros = y_hat - y
        return np.sum(erros**2, axis=0) * (1 / (2 * len(X)))

    def dloss(X, w, y):
        y_hat = linear_regression(X, w)
        errors = y_hat - y
        return (1 / len(X)) * X.T @ errors

    space = np.linspace(0.5, 5, 100)
    W1, W2 = np.meshgrid(space, space)

    stacked = np.stack([W1.flatten(), W2.flatten()], axis=0)

    batch_sizes = [len(X), 5, 20]

    for batch_size in batch_sizes:
        updates = 400
        alpha = 0.5
        init_w = np.array([[1], [1]])
        steps, losses = gradient_descent(
            loss, dloss, X, y, alpha, updates, init_w, batch_size
        )
        _, axes = plt.subplots(1, 2)
        ax1, ax2 = axes

        ax1.set_title(f'GD Trajectory Batch Size: {batch_size}')
        ax1.set_xlabel('w1')
        ax1.set_ylabel('w2')

        ax1.contour(
            W1, W2, loss(X, stacked, y).reshape(W1.shape), levels=20, cmap='plasma'
        )
        ax1.plot(*zip(*steps[::20]), color='blue', marker='o', label='Trajectory')
        ax1.scatter(*steps[-1], c='blue', label='Last Step')
        ax1.scatter(*W_star, c='red', s=30, label='Optimum')
        ax1.legend()

        ax2.set_title('GD Loss vs. Iterations')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Loss (log)')
        ax2.plot(range(updates + 1), losses, color='blue', label='Loss')
        ax2.set_yscale('symlog')
        ax2.legend()

        plt.savefig(f'./images/question1b_batch_size-{batch_size}.png')
        plt.show()


def question_2():
    pass


def question_3():
    pass


if __name__ == '__main__':
    if not os.path.exists('./images'):
        os.makedirs('./images')
    import matplotlib.pyplot as plt

    sp.init_printing(use_unicode=True, use_latex=False)

    question_1a()
    question_1b()

    question_2()
    question_3()
