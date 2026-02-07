import os
import random
from typing import List

import numpy as np
from IPython.display import display
from matplotlib import pyplot as plt
from sympy import Symbol, diff, init_printing, lambdify, symbols

init_printing(use_unicode=True, use_latex=False)


def gradient_descent(func, args: List[Symbol], a=0.05, init_vals=None, iters=50):
    if init_vals != None and len(args) != len(init_vals):
        raise Exception('Initial values have to be proivided for all args')

    derivatives = [
        lambdify(
            arg,
            diff(func, arg),
        )
        for arg in args
    ]
    xs: List = [random.uniform(0, 2)] * len(args) if init_vals == None else init_vals

    xs_steps = [xs]
    for _ in range(iters):
        xs = [x - a * d(x) for x, d in zip(xs, derivatives)]
        xs_steps.append(xs)

    return xs, xs_steps


def question_1():
    x = symbols('x', real=True)
    f_sym = x**4
    dfdx = diff(f_sym, x)
    display(f_sym, dfdx)

    ### part b
    f_sym = lambdify(x, f_sym)
    dfdx = lambdify(x, dfdx)
    delta = 0.01
    finite_diff_approx = lambda x, d: (f_sym(x + d) - f_sym(x)) / d

    x_range = np.arange(-2, 2.1, 0.1)

    plt.figure(figsize=(8, 5))
    plt.plot(x_range, dfdx(x_range), 'k--', label='Exact Derivative', linewidth=2.5)
    plt.plot(
        x_range,
        finite_diff_approx(x_range, delta),
        label=f'Finite difference (δ={0.01})',
    )

    plt.title('Exact Derivative vs. Forward Finite-Difference Approx.')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('Derivative', fontsize=14)
    plt.legend(loc='best', framealpha=0.95)
    plt.grid(visible=True, alpha=0.4)
    plt.savefig('./images/question1.b.png', dpi=300, bbox_inches='tight')
    plt.show()

    ### part c
    deltas = [0.001, 0.01, 0.05, 0.1, 0.5, 1]

    def mae(expected, actual):
        expected = np.array(expected)
        actual = np.array(actual)
        return np.mean(np.abs(expected - actual))

    mean_absolute_errors = []
    for d in deltas:
        expected = dfdx(x_range)
        actual = finite_diff_approx(x_range, d)
        mean_absolute_errors.append(mae(expected, actual))

    plt.plot(deltas, mean_absolute_errors, marker='o', label='MAE')
    plt.title('MAE for different δ', fontsize=16, pad=10)
    plt.xlabel('δ', fontsize=14)
    plt.xscale('log')
    plt.ylabel('MAE', fontsize=14)
    plt.grid(visible=True, alpha=0.4)
    plt.tight_layout()
    plt.legend(loc='best', framealpha=0.95)
    plt.savefig('./images/question1c.png', dpi=300, bbox_inches='tight')
    plt.show()

    ### part d
    x = symbols('x')
    f_sym = x**4

    def gradient_descent(function, arg, iters=20, alpha=0.05, init_val=1):

        dfdx = lambdify(arg, diff(function, arg))
        x = init_val
        steps = [x]

        for _ in range(iters):
            x -= alpha * dfdx(x)
            steps.append(x)

        return x, steps

    alphas = [0.05, 0.5, 1.2]
    _, axes = plt.subplots(1, len(alphas), figsize=(18.5, 8))
    axes[0].set_ylabel('f(x) / x', fontsize=14)

    f = lambdify(x, f_sym)

    for alpha, ax in zip(alphas, axes):
        iters = 20 if alpha <= 1 else 5
        x_range = range(iters + 1)

        _, steps = gradient_descent(f_sym, x, iters, alpha=alpha, init_val=1)

        steps = np.array(steps)

        ax.plot(x_range, f(steps), label='f(x)', marker='o')
        ax.set_title(f'f(x) and x vs. Iterations, {alpha=}', fontsize=14)
        ax.set_xlabel('Iterations', fontsize=14)
        if alpha >= 1:
            ax.set_yscale('symlog')

        ax.grid(visible=True, alpha=0.4)

    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=14)
    plt.savefig('./images/question1e.png', dpi=300, bbox_inches='tight')
    plt.show()


def question_2():
    x_0, x_1 = symbols('x_0 x_1')
    f_sym = 0.5 * (x_0**2 + 10 * x_1**2)

    f = lambdify(args=[x_0, x_1], expr=f_sym)

    value_range = np.linspace(-2, 2, 100)
    X_0, X_1 = np.meshgrid(value_range, value_range)

    cs = plt.contour(X_0, X_1, f(X_0, X_1), levels=10, cmap='plasma')
    plt.clabel(cs, inline=True, fontsize=8)
    plt.title(f'Countour Plot for f(x_0, x_1)={f_sym}', fontsize=14, pad=10)
    plt.xlabel('x_0', fontsize=14)
    plt.ylabel('x_1', fontsize=14)
    plt.savefig('./images/question2a.png', dpi=300, bbox_inches='tight')
    plt.show()

    ### part b
    alphas = [0.05, 0.2]
    _, axes = plt.subplots(1, 2, figsize=(15, 7))

    for alpha, ax in zip(alphas, axes):
        opt, steps = gradient_descent(
            f_sym, [x_0, x_1], a=alpha, init_vals=[1.5, 1.5], iters=100
        )
        cs = ax.contour(X_0, X_1, f(X_0, X_1), levels=10, cmap='plasma')
        ax.clabel(cs, inline=True, fontsize=8)
        ax.set_title(
            f'GD with {alpha=}, Optimum:({opt[0]:.2f}, {opt[1]:.2f}),',
            fontsize=14,
            pad=15,
        )
        ax.set_xlabel('x_0', fontsize=14)
        ax.set_ylabel('x_1', fontsize=14)

        ax.plot(*zip(*steps), 'b.-', linewidth=1, label=f'Gradient Descent')

    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1), fontsize=14)
    plt.savefig('./images/question2b.png', dpi=300, bbox_inches='tight')
    plt.show()

    # part II
    x = symbols('x')
    f_sym = x**4 - 2 * x**2 + 0.1 * x
    f = lambdify(x, f_sym)

    x_values = np.linspace(-2, 2, 100)

    x_opt, steps = gradient_descent(f_sym, [x], init_vals=[1.5], a=0.05)
    x_opt = x_opt[0]

    x_opt_2, steps_2 = gradient_descent(f_sym, [x], init_vals=[-1.5], a=0.05)
    x_opt_2 = x_opt_2[0]

    steps = np.array(steps)
    steps_2 = np.array(steps_2)

    plt.step(
        steps,
        f(steps),
        where='pre',
        color='red',
        linewidth='1',
        label='GD from x_0=1.5',
    )
    plt.plot(steps, f(steps), 'o', color='red', markersize=4)

    plt.step(
        steps_2,
        f(steps_2),
        where='pre',
        color='blue',
        linewidth='1',
        label='GD from x_0=-1.5',
    )
    plt.plot(steps_2, f(steps_2), 'o', color='blue', markersize=4)

    plt.plot(x_values, f(x_values), color='black')

    plt.title('GD comparision for x_0 = 1.5 vs. -1.5', fontsize=14)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('f(x)', fontsize=14)
    plt.legend(loc='best')
    plt.savefig('./images/question2c.png', dpi=300, bbox_inches='tight')
    plt.show()


def question_3():
    x = symbols('x')
    f_sym = x**2
    f = lambdify(x, f_sym)
    max_iters = 20

    alphas = [0.1, 0.01, 1.01]
    _, axes = plt.subplots(1, len(alphas), figsize=(16, 5))
    axes[0].set_ylabel('f(x) / x', fontsize=14)

    for alpha, ax in zip(alphas, axes):
        _, steps = gradient_descent(f_sym, [x], a=alpha, init_vals=[1], iters=max_iters)
        xs = np.array(steps).flatten()

        x_vals = range(max_iters + 1)
        ax.plot(x_vals, f(xs), label='f(x)')
        ax.plot(x_vals, xs, label='x')

        ax.set_title(f'f(x) and x vs. Iterations, alpha={alpha}', fontsize=14)
        ax.set_xlabel('Iterations', fontsize=14)
        ax.legend(loc='best')
        if alpha == 1.01:
            ax.set_yscale('symlog')
        else:
            ax.set_ylim(1e-4, 1e1)
            ax.set_yscale('log')

    plt.savefig(f'./images/question3b.png', dpi=300, bbox_inches='tight')
    plt.show()

    ### part2
    gammas = [0.5, 1, 2, 5]
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    max_iters = 20

    for gamma, ax in zip(gammas, axes.flatten()):
        x = symbols('x')
        f_sym = gamma * x**2
        f = lambdify(args=[x], expr=f_sym)

        _, steps = gradient_descent(f_sym, [x], a=0.1, init_vals=[1], iters=max_iters)

        xs = np.array(steps).flatten()

        x_vals = range(max_iters + 1)
        ax.plot(
            x_vals,
            f(xs),
            label='f(x)',
        )
        ax.plot(
            x_vals,
            xs,
            label='x',
        )
        ax.set_title(f'f(x) and x vs. Iterations, gamma {gamma}', fontsize=14)
        if gamma == 5:
            ax.set_yscale('symlog')
        else:
            ax.set_yscale('log')
            ax.set_ylim(1e-9, 1e1)

        ax.set_xlabel('Iterations')
        ax.set_ylabel('f(x) / x')
        ax.legend(loc='best')

    plt.savefig(f'./images/question3d.png', dpi=300, bbox_inches='tight')
    plt.show()

    ### part 3
    max_iters = 60
    alpha = 0.1
    initial_value = 1.0

    x = initial_value
    steps = [x]
    for _ in range(max_iters):
        step = alpha * np.sign(x)
        x -= step
        steps.append(x)

    xs = range(max_iters + 1)

    plt.plot(xs, steps, label='x')
    plt.plot(xs, np.abs(steps), label='f(x)')

    plt.title(f'f(x)/x vs. Iterations for {f_sym}', fontsize=14)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('f(x) / x', fontsize=14)
    plt.legend(loc='best', framealpha=0.95)
    plt.savefig('./images/question3f.png', dpi=300, bbox_inches='tight')
    plt.show()


def question_4_a_b():
    gammas = [1, 4]
    alpha = 0.1
    initial_value = 1
    val_range = np.linspace(-1.5, 1.5, 100)
    X1, X2 = np.meshgrid(val_range, val_range)

    x1, x2 = symbols('x1 x2')

    _, axes = plt.subplots(1, 2, figsize=(20, 8))
    for gamma, ax in zip(gammas, axes):
        f_sym = x1**2 + gamma * x2**2
        f = lambdify([x1, x2], f_sym)
        cs = ax.contour(X1, X2, f(X1, X2), levels=10, cmap='plasma')
        plt.clabel(cs, inline=True, fontsize=8)
        ax.set_title(f'Contour Plot for {f_sym}, Gamma={gamma}', fontsize=14)
        ax.set_xlabel('X1', fontsize=14)
        ax.set_ylabel('X2', fontsize=14)

    plt.savefig('./images/question4a.png', dpi=300, bbox_inches='tight')
    plt.show()

    _, axes = plt.subplots(1, 2, figsize=(20, 8))
    for gamma, ax in zip(gammas, axes):
        f_sym = x1**2 + gamma * x2**2
        f = lambdify([x1, x2], f_sym)
        cs = ax.contour(X1, X2, f(X1, X2), levels=10, cmap='plasma')
        plt.clabel(cs, inline=True, fontsize=8)

        _, steps = gradient_descent(
            f_sym, [x1, x2], alpha, [initial_value, initial_value]
        )
        ax.plot(*zip(*steps), 'b.-', linewidth=1, label='Gradient Decent Steps')

        ax.set_title(f'Contour Plot for {f_sym}, Gamma={gamma}', fontsize=14)
        ax.set_xlabel('X1', fontsize=14)
        ax.set_ylabel('X2', fontsize=14)
        ax.legend(loc='upper right', framealpha=0.95, fontsize='x-large')

    plt.savefig('./images/question4b.png', dpi=300, bbox_inches='tight')
    plt.show()


def question_4_c_d():
    x1, x2 = symbols('x1 x2')
    f_sym = (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2

    dfdx1 = diff(f_sym, x1)
    dfdx1 = lambdify([x1, x2], dfdx1)

    dfdx2 = diff(f_sym, x2)
    dfdx2 = lambdify([x1, x2], dfdx2)

    f = lambdify([x1, x2], f_sym)
    X1, X2 = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-1, 3, 50))

    # part c
    plt.figure(figsize=(12, 8))
    cs = plt.contour(X1, X2, f(X1, X2), levels=25, cmap='plasma')
    plt.clabel(cs, inline=True, fontsize=6)
    plt.title(f'Contour Plot of f(x_1, x_2)={f_sym}', fontsize=14)
    plt.xlabel('X1', fontsize=14)
    plt.ylabel('X2', fontsize=14)
    plt.savefig('./images/question4c.png', dpi=300, bbox_inches='tight')
    plt.show()

    # part d
    alphas = [0.001, 0.005]
    max_iters = 2000
    _, axes = plt.subplots(1, 2, figsize=(20, 6))

    for alpha, ax in zip(alphas, axes):
        ax.contour(X1, X2, f(X1, X2), levels=25, cmap='plasma')

        xs = [-1.25, 0.5]
        steps = [xs]

        for _ in range(max_iters):
            xs = [x - alpha * d(*xs) for x, d in zip(xs, [dfdx1, dfdx2])]
            steps.append(xs)

        ax.plot(*zip(*steps), 'r.-', linewidth=1)
        ax.set_title(
            f'GD with {alpha=}, Optimum=({xs[0]:.3f}, {xs[1]:.3f})', fontsize=14
        )
        ax.set_xlabel('X1', fontsize=14)
        ax.set_ylabel('X2', fontsize=14)

    plt.savefig('./images/question4d.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    if not os.path.exists('./images'):
        os.makedirs('./images')

    question_1()
    question_2()
    question_3()
    question_4_a_b()
    question_4_c_d()
