import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from numpy.linalg import inv
from numpy.typing import NDArray

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


def constant_optimizer(alpha, grad_fn):
    return lambda X, w, y: -alpha * grad_fn(X, w, y)


def adagrad(alpha_0, grad_fn) -> Callable:
    alpha_t = alpha_0
    sum = 0

    def step_fn(X, w, y):
        nonlocal alpha_t, sum
        grads = grad_fn(X, w, y)

        sum += grads**2

        alpha_t = -alpha_0 / np.sqrt(sum + 0.000001)

        return alpha_t * grads

    return step_fn


def nag(alpha, beta, grad_fn) -> Callable:
    z = 0

    def step_fn(X, w, y):
        nonlocal z

        z = beta * z - alpha * grad_fn(X, (w + beta * z), y)
        return z

    return step_fn


def gradient_descent(
    loss_fn: Callable,
    X: NDArray,
    y: NDArray,
    optimizer: Callable,
    max_updates: int,
    init_w,
    batch_size=None,
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
            if number_updates >= max_updates:
                break

            X_batch = X_shuff[batch : batch + batch_size]
            y_batch = y_shuff[batch : batch + batch_size]

            step = optimizer(X_batch, w, y_batch)
            w = w + step

            number_updates += 1
            losses.append(loss_fn(X, w, y).item())
            steps.append(w)

    return steps, losses


def _setup_question1():
    rng = np.random.default_rng(67)

    m = 1000
    X = rng.normal(0.0, 1.0, size=(m, 2))
    noise = rng.normal(0.0, 1.0, size=(m, 1))
    W_star = np.array([3, 4])

    linear_regression = lambda X, w: (X @ w).reshape(X.shape[0], -1)
    y = linear_regression(X, W_star) + noise

    def loss_fn(X, w, y):
        y_hat = linear_regression(X, w)
        errors = y_hat - y
        return np.sum(errors**2, axis=0) * (1 / (2 * len(X)))

    def dloss_fn(X, w, y):
        y_hat = linear_regression(X, w)
        errors = y_hat - y
        return (1 / len(X)) * X.T @ errors

    space = np.linspace(0.5, 5, 100)
    W1, W2 = np.meshgrid(space, space)
    W_combinations = np.stack([W1.flatten(), W2.flatten()], axis=0)

    return X, y, W_star, loss_fn, dloss_fn, W1, W2, W_combinations


def question_1a():
    X, y, W_star, loss_fn, dloss_fn, W1, W2, W_combinations = _setup_question1()

    epochs = 80
    alpha = 0.5
    init_w = np.array([[1], [1]])

    optimizer = constant_optimizer(alpha, dloss_fn)
    steps, losses = gradient_descent(loss_fn, X, y, optimizer, epochs, init_w)

    _, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    ax1.set_title('Full GD Trajectory')
    ax1.set_xlabel('w1')
    ax1.set_ylabel('w2')

    ax1.contour(
        W1,
        W2,
        loss_fn(X, W_combinations, y).reshape(W1.shape),
        levels=30,
        cmap='plasma',
    )
    ax1.plot(*zip(*steps), color='blue', marker='o', label='Trajectory')
    ax1.scatter(*steps[-1], c='blue', label='Last Step')
    ax1.scatter(*W_star, c='black', s=30, label='Optimum')
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
    X, y, W_star, loss_fn, dloss_fn, W1, W2, W_combinations = _setup_question1()

    batch_sizes = [len(X), 5, 20]
    colors = ['blue', 'red', 'green']
    updates = 400
    alpha = 0.5

    all_steps = []
    all_losses = []
    optimizer = constant_optimizer(alpha, dloss_fn)
    for batch_size in batch_sizes:
        init_w = np.array([[1], [1]])
        steps, batch_losses = gradient_descent(
            loss_fn, X, y, optimizer, updates, init_w, batch_size
        )
        all_steps.append(steps)
        all_losses.append(batch_losses)

    fig_contour, ax_contour = plt.subplots()
    ax_contour.set_title('SGD Trajectories by Batch Size')
    ax_contour.set_xlabel('w1')
    ax_contour.set_ylabel('w2')
    ax_contour.contour(
        W1,
        W2,
        loss_fn(X, W_combinations, y).reshape(W1.shape),
        levels=20,
        cmap='plasma',
    )
    for steps, batch_size, color in zip(all_steps, batch_sizes, colors):
        ax_contour.plot(
            *zip(*steps[::10]),
            color=color,
            marker='o',
            label=f'Batch size {batch_size}',
        )
        ax_contour.scatter(*steps[-1], c=color)
    ax_contour.scatter(*W_star, c='black', s=30, label='Optimum')
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
    X, y, W_star, loss_fn, dloss_fn, W1, W2, W_combinations = _setup_question1()

    updates = 400
    batch_size = 20
    alpha = 0.5
    beta = 0.9

    optimizers = [
        ('Constant', constant_optimizer(alpha, dloss_fn), 'blue'),
        ('NAG', nag(alpha, beta, dloss_fn), 'red'),
        ('Adagrad', adagrad(alpha, dloss_fn), 'green'),
    ]

    all_results = []
    for name, optimizer, color in optimizers:
        init_w = np.array([[1], [1]])
        steps, losses = gradient_descent(
            loss_fn, X, y, optimizer, updates, init_w, batch_size
        )
        all_results.append((name, steps, losses, color))

    fig_contour, ax_contour = plt.subplots()
    ax_contour.set_title(f'Optimizer Trajectories (batch size {batch_size})')
    ax_contour.set_xlabel('w1')
    ax_contour.set_ylabel('w2')
    ax_contour.contour(
        W1,
        W2,
        loss_fn(X, W_combinations, y).reshape(W1.shape),
        levels=20,
        cmap='plasma',
    )

    for name, steps, _, color in all_results:
        ax_contour.plot(
            *zip(*steps[::10]), color=color, marker='o', linewidth=1.5, label=name
        )
        ax_contour.scatter(*steps[-1], c=color)

    ax_contour.scatter(*W_star, c='black', s=30, label='Optimum')
    ax_contour.legend()
    ax_contour.grid(False)
    fig_contour.savefig('./images/question1c_trajectories.png')
    plt.show()

    fig_loss, ax_loss = plt.subplots()
    ax_loss.set_title(f'Loss vs. Updates by Optimizer (batch size {batch_size})')
    ax_loss.set_xlabel('Updates')
    ax_loss.set_ylabel('Loss (log)')

    for name, _, losses, color in all_results:
        ax_loss.plot(range(updates + 1), losses, color=color, label=name)

    ax_loss.set_yscale('symlog')
    ax_loss.legend()
    fig_loss.savefig('./images/question1c_loss.png')
    plt.show()


def _setup_question2():
    rng = np.random.default_rng(67)

    m = 1000
    U = rng.uniform(-2, 2, size=(m, 1))
    noise = rng.normal(0, 0.05, size=(m, 1))
    X_star = np.array([1, 3])

    space = np.linspace(0, 3.5)
    X1, X2 = np.meshgrid(space, space)
    X_combinations = np.stack([X1.flatten(), X2.flatten()], axis=0)

    def h(U, x):
        x_1, x_2 = x
        return x_2 * np.tanh(x_1 * U)

    def loss_fn(U, x, y):
        y_hat = h(U, x)
        errors = y_hat - y

        return (1 / len(U)) * np.sum(0.5 * errors**2, axis=0)

    def dloss_fn(U, x, y):
        x_1, x_2 = x
        y_hat = h(U, x)
        error = y_hat - y

        djdx1 = np.mean(error * x_2 * (1 - np.tanh(x_1 * U) ** 2) * U, axis=0)
        djdx2 = np.mean(error * np.tanh(x_1 * U), axis=0)

        return np.array([djdx1, djdx2])

    y = h(U, X_star) + noise

    return U, y, X_star, loss_fn, dloss_fn, X1, X2, X_combinations


def question_2a():
    U, y, X_star, loss_fn, dloss_fn, X1, X2, X_comb = _setup_question2()

    max_updates = 500
    optimizer = constant_optimizer(alpha=0.75, grad_fn=dloss_fn)

    steps, losses = gradient_descent(
        loss_fn, U, y, optimizer, max_updates, np.array([[1], [1]])
    )
    _, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    ax1.set_title('Full GD Trajectory')
    ax1.set_xlabel('w1')
    ax1.set_ylabel('w2')

    ax1.contour(
        X1,
        X2,
        loss_fn(U, X_comb, y).reshape(X1.shape),
        levels=20,
        cmap='plasma',
    )
    ax1.plot(*zip(*steps), color='blue', marker='o', label='Trajectory')
    ax1.scatter(*X_star, c='red', s=30, label='Optimum')
    ax1.legend()

    ax2.set_title('Full GD Loss vs. Iterations')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Loss (log)')
    ax2.plot(range(max_updates + 1), losses, color='blue', label='Loss')
    ax2.set_yscale('log')
    ax2.legend()

    plt.savefig('./images/quertion2a.png')
    plt.show()


def question_2b():
    U, y, X_star, loss_fn, dloss_fn, X1, X2, X_comb = _setup_question2()

    batch_sizes = [len(U), 20, 5]
    colors = ['red', 'green', 'blue']
    colors = ['blue', 'red', 'green']
    updates = 500
    alpha = 0.5

    all_steps = []
    all_losses = []
    optimizer = constant_optimizer(alpha, dloss_fn)
    for batch_size in batch_sizes:
        init_w = np.array([[1], [1]])
        steps, batch_losses = gradient_descent(
            loss_fn, U, y, optimizer, updates, init_w, batch_size
        )
        all_steps.append(steps)
        all_losses.append(batch_losses)

    fig_contour, ax_contour = plt.subplots(figsize=(10, 8))
    ax_contour.set_title('SGD Trajectories by Batch Size')
    ax_contour.set_xlabel('w1')
    ax_contour.set_ylabel('w2')
    ax_contour.contour(
        X1,
        X2,
        loss_fn(U, X_comb, y).reshape(X1.shape),
        levels=20,
        cmap='plasma',
    )
    for steps, batch_size, color in zip(all_steps, batch_sizes, colors):
        ax_contour.plot(
            *zip(*steps[::]),
            color=color,
            marker='o',
            label=f'Batch size {batch_size}',
        )
        ax_contour.scatter(*steps[-1], c=color)
    ax_contour.scatter(*X_star, c='black', s=30, label='Optimum')
    ax_contour.legend()
    fig_contour.savefig('./images/question2b_trajectories.png')
    plt.show()

    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    for batch_losses, batch_size, color in zip(all_losses, batch_sizes, colors):
        ax_loss.set_title(f'SGD Loss vs. Updates (Batch size {batch_size})')
        ax_loss.set_xlabel('Updates')
        ax_loss.set_ylabel('Loss (log)')
        ax_loss.plot(
            range(updates + 1),
            batch_losses,
            color=color,
            label=f'Batch size {batch_size}',
        )
        ax_loss.set_yscale('symlog')
        ax_loss.legend()
    fig_loss.savefig(f'./images/question2b_loss.png')
    plt.show()


def question_2c():
    U, y, X_star, loss_fn, dloss_fn, X1, X2, X_comb = _setup_question2()

    updates = 500
    batch_size = 10
    alpha = 0.5
    beta = 0.9

    optimizers = [
        ('Constant', constant_optimizer(alpha, dloss_fn), 'blue'),
        ('NAG', nag(alpha, beta, dloss_fn), 'red'),
        ('Adagrad', adagrad(alpha, dloss_fn), 'green'),
    ]

    all_results = []
    for name, optimizer, color in optimizers:
        init_w = np.array([[1], [1]])
        steps, losses = gradient_descent(
            loss_fn, U, y, optimizer, updates, init_w, batch_size
        )
        all_results.append((name, steps, losses, color))

    fig_contour, ax_contour = plt.subplots()
    ax_contour.set_title(f'Optimizer Trajectories (batch size {batch_size})')
    ax_contour.set_xlabel('w1')
    ax_contour.set_ylabel('w2')
    ax_contour.contour(
        X1,
        X2,
        loss_fn(U, X_comb, y).reshape(X1.shape),
        levels=20,
        cmap='plasma',
    )

    for name, steps, _, color in all_results:
        ax_contour.plot(*zip(*steps[::]), color=color, marker='o', label=name)
        ax_contour.scatter(*steps[-1], c=color)

    ax_contour.scatter(*X_star, c='black', s=30, label='Optimum')
    ax_contour.legend()
    ax_contour.grid(False)
    fig_contour.savefig('./images/question2c_trajectories.png')
    plt.show()

    fig_loss, ax_loss = plt.subplots()
    ax_loss.set_title(f'Loss vs. Updates by Optimizer (batch size {batch_size})')
    ax_loss.set_xlabel('Updates')
    ax_loss.set_ylabel('Loss (log)')

    for name, _, losses, color in all_results:
        ax_loss.plot(range(updates + 1), losses, color=color, label=name)

    ax_loss.set_yscale('symlog')
    ax_loss.legend()
    fig_loss.savefig('./images/question2c_loss.png')
    plt.show()


def _setup_question3():
    x1, x2 = sp.symbols('x1 x2')
    rosenbrock = (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2
    f = sp.lambdify([x1, x2], rosenbrock)

    dfdx1 = sp.lambdify([x1, x2], sp.diff(rosenbrock, x1))
    dfdx2 = sp.lambdify([x1, x2], sp.diff(rosenbrock, x2))
    grad_fn = lambda x: np.array([dfdx1(*x), dfdx2(*x)])

    hessian_fn = sp.lambdify([x1, x2], sp.hessian(rosenbrock, (x1, x2)))

    return f, grad_fn, hessian_fn


def gd(f, x, grad_fn, alpha, iters):
    steps = [x.copy()]
    losses = [f(*x)]
    for _ in range(iters):
        grads = grad_fn(x)
        x = x - alpha * grads

        steps.append(x.copy())
        losses.append(f(*x))

    return steps, losses


def newtons_update_gd(f, x, grad_fn, hessian_fn, alpha, iters):
    steps = [x.copy()]
    losses = [f(*x)]
    lam = 1e-6
    for _ in range(iters):
        x = x - alpha * inv(hessian_fn(*x) + lam * np.eye(2, 2)) @ grad_fn(x)

        steps.append(x.copy())
        losses.append(f(*x))

    return steps, losses


def damped_newton_step(fn, x, grad_fn, hessian_fn, alpha_0, max_k, shrink_factor):
    fx_curr = fn(*x)
    grads = grad_fn(x)
    hessian = hessian_fn(*x)
    lam = 1e-6

    alpha = alpha_0
    for _ in range(max_k):
        step = alpha * inv(hessian + lam * np.eye(2, 2)) @ grads
        x_new = x - step

        fx_new = fn(*x_new)
        if fx_new < fx_curr:
            return step

        alpha *= shrink_factor

    return 10**-4 * grads


def damped_newton_gd(fn, x, grad_fn, hessian_fn, alpha_0, iters, max_k, shrink_factor):
    steps = [x.copy()]
    losses = [fn(*x)]
    for _ in range(iters):
        step = damped_newton_step(
            fn, x, grad_fn, hessian_fn, alpha_0, max_k, shrink_factor
        )
        x -= step

        steps.append(x.copy())
        losses.append(fn(*x))

    return steps, losses


def question_3a():
    f, grad_fn, _ = _setup_question3()

    alpha = 10**-3
    iterations = 200
    x = np.array([-1.0, 1.0])
    _, losses = gd(f, x, grad_fn, alpha, iterations)

    min_idx = int(np.argmin(losses))

    plt.figure(figsize=(12, 8))
    plt.plot(losses, color='blue', label='f(x)')
    plt.scatter(
        min_idx,
        losses[min_idx],
        color='red',
        zorder=5,
        label=f'Min: f={losses[min_idx]:.4f} at iter {min_idx}',
    )
    plt.title('Rosenbrock GD: f(x) vs Iterations', fontsize=14)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('f(x) (log)', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=9)
    plt.savefig('./images/question3a.png', dpi=300, bbox_inches='tight')
    plt.show()


def question_3b():
    f, grad_fn, hessian_fn = _setup_question3()

    alpha = 0.7
    iters = 200
    x = np.array([-1.0, 1.0])

    _, losses = newtons_update_gd(f, x, grad_fn, hessian_fn, alpha, iters)

    min_idx = int(np.argmin(losses))

    plt.figure(figsize=(12, 8))
    plt.plot(losses, color='blue', label='f(x)')
    plt.scatter(
        min_idx,
        losses[min_idx],
        color='red',
        zorder=5,
        label=f'Min: f={losses[min_idx]:.4f} at iter {min_idx}',
    )
    plt.title('Rosenbrock Newton Updates: f(x) vs Iterations', fontsize=14)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('f(x) (log)', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=9)
    plt.savefig('./images/question3b.png', dpi=300, bbox_inches='tight')
    plt.show()


def question_3c():
    f, grad_fn, hessian_fn = _setup_question3()

    x0 = np.array([-1.0, 1.0])
    iterations = 200

    gd_steps, gd_losses = gd(f, x0, grad_fn, alpha=1e-3, iters=iterations)
    newton_steps, newton_losses = newtons_update_gd(
        f, x0, grad_fn, hessian_fn, alpha=0.7, iters=iterations
    )

    damped_steps, damped_losses = damped_newton_gd(
        f,
        x0,
        grad_fn,
        hessian_fn,
        alpha_0=0.7,
        iters=iterations,
        max_k=20,
        shrink_factor=0.5,
    )

    gd_steps = np.array(gd_steps)
    newton_steps = np.array(newton_steps)
    damped_steps = np.array(damped_steps)

    plt.figure(figsize=(12, 8))
    plt.plot(gd_losses, label='GD', color='red')
    plt.plot(newton_losses, label='Newton', color='green')
    plt.plot(damped_losses, label='Damped Newton', color='blue')
    plt.title('Rosenbrock: f(x) vs Iterations', fontsize=14)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('f(x) (log)', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=9)
    plt.savefig('./images/question3c_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

    X1, X2 = np.meshgrid(np.linspace(-3, 2, 100), np.linspace(-2, 2.5, 100))
    plt.figure(figsize=(12, 8))

    cs = plt.contour(X1, X2, f(X1, X2), levels=30, cmap='plasma')
    plt.clabel(cs, inline=True, fontsize=6)
    plt.plot(
        gd_steps[:, 0],
        gd_steps[:, 1],
        linewidth=1.5,
        label='GD',
        color='red',
        marker='o',
        markersize=3,
    )
    plt.plot(
        newton_steps[:, 0],
        newton_steps[:, 1],
        linewidth=1.5,
        label='Newton',
        color='green',
        marker='o',
        markersize=3,
    )
    plt.plot(
        damped_steps[:, 0],
        damped_steps[:, 1],
        linewidth=1.5,
        label='Damped Newton',
        color='blue',
        marker='o',
        markersize=3,
    )
    plt.scatter(*x0, c='white', s=40, zorder=5, label='Start')
    plt.title('Rosenbrock: Trajectories', fontsize=14)
    plt.xlabel('x1', fontsize=14)
    plt.ylabel('x2', fontsize=14)
    plt.legend(fontsize=9)
    plt.savefig('./images/question3c_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    if not os.path.exists('./images'):
        os.makedirs('./images')

    sp.init_printing(use_unicode=True, use_latex=False)

    # question_1a()
    # question_1b()
    # question_1c()
    #
    # question_2a()
    # question_2b()
    # question_2c()
    #
    question_3a()
    question_3b()
    question_3c()
