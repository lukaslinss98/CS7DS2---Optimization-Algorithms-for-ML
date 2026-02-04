import os
import random
from typing import Callable, List

import numpy as np
from matplotlib import pyplot as plt
from sympy import Abs, Symbol, diff, init_printing, lambdify, symbols

init_printing(pretty_print=True, use_latex=False)


def gradient_descent(func, args: List[Symbol], a=0.05, initial_vals=None, iters=50):
    if initial_vals != None and len(args) != len(initial_vals):
        raise Exception('Initial values have to be proivided for all args')

    derivatives = [
        lambdify(
            arg,
            diff(func, arg),
        )
        for arg in args
    ]
    xs: List = (
        [random.uniform(0, 2)] * len(args) if initial_vals == None else initial_vals
    )

    xs_steps = [xs]
    for _ in range(iters):
        xs = [x - a * d(x) for x, d in zip(xs, derivatives)]
        xs_steps.append(xs)

    return xs, xs_steps


def gradient_descent_foo(derivatives: List, args: List, a=0.05, initial=1.5, iters=50):
    xs = [initial] * len(args)

    xs_walk = [xs]
    for _ in range(iters):
        xs = [x - a * d(x) for x, d in zip(xs, derivatives)]
        xs_walk.append(xs)

    return xs, xs_walk


def question_1():
    x = symbols('x', real=True)
    f = x**4
    dfdx = diff(f, x)
    print(dfdx)

    ### part b
    f = lambdify(x, f)
    dfdx = lambdify(x, dfdx)

    x_values = np.arange(-2, 2.1, 0.1)
    y_values = [dfdx(x) for x in x_values]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, 'k--', label='Exact Derivative', linewidth=2.5)

    finite_diff_approx = lambda f, x, delta: (f(x + delta) - f(x)) / delta

    y_values_approx = [finite_diff_approx(f, xs, delta=0.01) for xs in x_values]

    plt.plot(x_values, y_values_approx, label=f'Finite difference (δ={0.01})')

    plt.title('Exact Derivative vs. Forward Finite-Difference Approx.')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('Derivative', fontsize=14)
    plt.legend(loc='best', framealpha=0.95)
    plt.savefig('./images/question1.b.png', dpi=300, bbox_inches='tight')
    plt.show()

    ### part c
    deltas = [0.001, 0.01, 0.05, 0.1, 0.5, 1]

    def mae(expected, actual):
        expected = np.array(expected)
        actual = np.array(actual)
        return np.mean(np.abs(expected - actual))

    mean_squared_errors = []
    for d in deltas:
        y_approx_values = [finite_diff_approx(f, x, d) for x in x_values]
        mean_squared_errors.append(mae(y_values, y_approx_values))

    plt.plot(deltas, mean_squared_errors, marker='o', label='MAE')
    plt.title('MAE for different δ', fontsize=16, pad=10)
    plt.xlabel('δ', fontsize=14)
    plt.ylabel('MAE', fontsize=14)
    plt.grid(visible=True, alpha=0.4)
    plt.tight_layout()
    plt.legend(loc='best', framealpha=0.95)
    plt.savefig('./images/question1c.png', dpi=300, bbox_inches='tight')
    plt.show()

    ### part d
    x = symbols('x')
    f = x**4

    def gradient_descent(function, args, iterations=100, alpha=0.05, initial_value=1):

        f = lambdify(args, function)
        dfdx = lambdify(args, diff(function, args))
        x = initial_value
        x_walk = [x]
        function_values = [f(x)]

        for iteration in range(1, iterations + 1):
            step = alpha * dfdx(x)
            x -= step
            x_walk.append(x)
            function_values.append(f(x))

            if iteration % 10 == 0:
                print(f'Iteration: {iteration}, value: {x}')

        plt.plot(range(iterations + 1), function_values, label='f(x)')
        plt.plot(range(iterations + 1), x_walk, label='x')
        plt.title(f'f(x) and x vs. iterations, {alpha=}')
        plt.xlabel('iterations')
        plt.ylabel('f(x) / x')
        plt.legend(loc='best', framealpha=0.95)
        plt.savefig(
            f'./images/question1e_alpha={alpha}.png', dpi=300, bbox_inches='tight'
        )
        plt.show()
        return x

    optimum = gradient_descent(f, x, iterations=20, alpha=0.05, initial_value=1)
    print(f'{optimum=}, {0.05=}')

    optimum = gradient_descent(f, x, iterations=20, alpha=0.5, initial_value=1)
    print(f'{optimum=}, {0.5=}')

    optimum = gradient_descent(f, x, iterations=5, alpha=1.2, initial_value=1)
    print(f'{optimum=}, {1.2=}')


### Question II
### part a
def question_2():
    x_0, x_1 = symbols('x_0 x_1')
    f_sym = 0.5 * (x_0**2 + 10 * x_1**2)
    dfdx_0 = diff(f_sym, x_0)
    dfdx_1 = diff(f_sym, x_1)

    f = lambdify(args=[x_0, x_1], expr=f_sym)

    print(dfdx_0, dfdx_1)
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

    _, xs_walk = gradient_descent(
        f_sym, [x_0, x_1], a=0.05, initial_vals=[1.5, 1.5], iters=100
    )
    _, xs_walk_2 = gradient_descent(
        f_sym, [x_0, x_1], a=0.02, initial_vals=[1.5, 1.5], iters=100
    )

    cs = plt.contour(X_0, X_1, f(X_0, X_1), levels=10, cmap='plasma')
    plt.clabel(cs, inline=True, fontsize=8)
    plt.title(f'Gradient Decent on f(x_0, x_1)={f_sym}', fontsize=14, pad=15)
    plt.xlabel('x_0', fontsize=14)
    plt.ylabel('x_1', fontsize=14)

    plt.plot(*zip(*xs_walk), 'b.-', linewidth=1, label='alpha=0.05')
    plt.plot(*zip(*xs_walk_2), 'r.-', linewidth=1, label='alpha=0.02')

    plt.legend(loc='best', framealpha=0.95)
    plt.savefig('./images/question2b.png', dpi=300, bbox_inches='tight')
    plt.show()

    # part I
    x = symbols('x')
    f_sym = x**4 - 2 * x**2 + 0.1 * x
    print(f_sym)
    f = lambdify(x, f_sym)

    x_values = np.linspace(-2, 2, 100)

    x_opt, _ = gradient_descent(f_sym, [x], initial_vals=[1.5], a=0.05)
    x_opt = x_opt[0]
    y_opt = f(x_opt)

    x_opt2, _ = gradient_descent(f_sym, [x], initial_vals=[-1.5], a=0.05)
    x_opt2 = x_opt2[0]
    y_opt2 = f(x_opt2)

    plt.plot(x_values, f(x_values))
    plt.scatter(
        x_opt,
        y_opt,
        marker='o',
        s=50,
        color='red',
        label=f'x_0=1.5 optimum ({x_opt:.4f},{y_opt:.4f})',
    )
    plt.scatter(
        x_opt2,
        y_opt2,
        marker='o',
        s=50,
        color='green',
        label=f'x_0=-1.5 optimum ({x_opt:.4f},{y_opt:.4f})',
    )
    plt.legend(loc='best', framealpha=0.95)
    plt.title('Optimum Comparision for x_0 1.5 vs. -1.5', fontsize=14)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('f(x)', fontsize=14)
    plt.savefig('./images/question2c.png', dpi=300, bbox_inches='tight')
    plt.show()


### Question 3
### part 1
def question_3():
    x = symbols('x')
    f_sym = x**2
    f = lambdify(x, f_sym)
    max_iters = 20
    alpha = 0.1

    _, steps = gradient_descent(f_sym, [x], a=0.1, initial_vals=[1], iters=max_iters)
    xs = np.array(steps).flatten()

    x_vals = range(max_iters + 1)
    plt.plot(x_vals, f(xs), label='f(x)')
    plt.plot(x_vals, xs, label='x')

    plt.title(f'f(x) and x vs. Iterations, alpha={alpha}', fontsize=14)
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('f(x) / x', fontsize=14)
    plt.legend(loc='best', framealpha=0.95)
    plt.savefig(f'./images/question3b_alpha{alpha}.png', dpi=300, bbox_inches='tight')
    plt.show()

    ### part2
    gammas = [0.5, 1, 2, 5]
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    for gamma, ax in zip(gammas, axes.flatten()):
        max_iters = 50

        x = symbols('x')
        f_sym = gamma * x**2
        f = lambdify(args=[x], expr=f_sym)

        _, steps = gradient_descent(f_sym, [x], a=0.1, initial_vals=[1], iters=50)

        xs = np.array(steps).flatten()

        x_vals = range(max_iters + 1)
        ax.plot(x_vals, f(xs), 'r--', label='f(x)', linewidth=1)
        ax.plot(x_vals, xs, label='x', linewidth=3)
        ax.set_title(f'f(x) and x vs. Iterations, gamma {gamma}', fontsize=14)
        ax.set_yscale('log')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('f(x) / x')
        ax.legend(loc='best')

    plt.savefig(f'./images/question3d.png', dpi=300, bbox_inches='tight')
    plt.show()

    ### part 3
    x = symbols('x')
    f_sym = Abs(x)
    max_iters = 60
    alpha = 0.1
    initial_value = 1

    _, steps = gradient_descent_foo(
        [np.sign], [x], a=alpha, iters=max_iters, initial=initial_value
    )
    xs = range(max_iters + 1)

    plt.plot(xs, steps, label='x')
    plt.plot(xs, np.abs(steps), label='f(x)')

    plt.title(f'f(x)/x vs. Iterations for {f_sym}', fontsize=14)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('f(x) / x', fontsize=14)
    plt.legend(loc='best', framealpha=0.95)
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
    plt.clabel(cs, inline=True, fontsize=8)
    plt.show()

    # part d
    alphas = [0.001, 0.005]
    max_iters = 2000
    _, axes = plt.subplots(1, 2, figsize=(20, 8))

    for alpha, ax in zip(alphas, axes):
        ax.contour(X1, X2, f(X1, X2), levels=25, cmap='plasma')

        xs = [-1.25, 0.5]
        steps = [xs]

        for _ in range(max_iters):
            xs = [x - alpha * d(*xs) for x, d in zip(xs, [dfdx1, dfdx2])]
            steps.append(xs)

        ax.plot(*zip(*steps), 'r.-', linewidth=1)
        ax.set_title(f'Gradient Descent on {f_sym}, {alpha=}', fontsize=14)
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
