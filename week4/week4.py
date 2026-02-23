import os

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from sympy import init_printing

init_printing(use_unicode=True, use_latex=False)


def polyak_step_size(fn, args, init_vals, iters=50):
    xs = np.array(init_vals)
    xs_steps = [init_vals]

    derivatives = [sp.diff(fn, arg) for arg in args]

    f = sp.lambdify(args, expr=fn)
    min_args = sp.solve(derivatives, args, dict=True)
    f_star = float(f(*list(min_args[0].values())))

    derivatives = [sp.lambdify(args, expr=d) for d in derivatives]
    for _ in range(iters):
        grads = np.array([d(*xs) for d in derivatives])

        alpha = (f(*xs) - f_star) / np.dot(grads, grads) + 0.00001
        xs = xs - alpha * grads
        xs_steps.append(xs.tolist())

    return xs, xs_steps


def rms_prop(fn, args, alpha, beta, init_vals, iters=50):
    a_0 = alpha
    xs = np.array(init_vals)
    xs_steps = [init_vals]

    derivatives = [sp.lambdify(args, sp.diff(fn, arg)) for arg in args]

    sum = 0
    for _ in range(iters):
        grads = np.array([d(*xs) for d in derivatives])
        sum = beta * sum + (1 - beta) * grads**2

        alpha = a_0 / (np.sqrt(sum) + 0.00001)
        xs = xs - alpha * grads
        xs_steps.append(xs.tolist())

    return xs, xs_steps


def heavy_ball(fn, args, alpha, beta, init_vals, iters=50):
    xs = np.array(init_vals)
    xs_steps = [init_vals]

    derivatives = [sp.lambdify(args, sp.diff(fn, arg)) for arg in args]

    z = np.zeros(len(args))
    for _ in range(iters):
        grads = np.array([d(*xs) for d in derivatives])

        z = beta * z + alpha * grads
        xs = xs - z
        xs_steps.append(xs.tolist())

    return xs, xs_steps


def adam(fn, args, alpha, beta_1, beta_2, init_vals, iters=50):
    xs = np.array(init_vals)
    xs_steps = [init_vals]

    derivatives = [sp.lambdify(args, sp.diff(fn, arg)) for arg in args]

    v = np.zeros(len(args))
    m = np.zeros(len(args))
    for t in range(1, iters + 1):
        grads = np.array([d(*xs) for d in derivatives])

        m = beta_1 * m + (1 - beta_1) * grads
        v = beta_2 * v + (1 - beta_2) * grads**2

        m_hat = m / (1 - beta_1**t)
        v_hat = v / (1 - beta_2**t)

        xs = xs - alpha * m_hat / (np.sqrt(v_hat) + 0.00001)
        xs_steps.append(xs.tolist())

    return xs, xs_steps


def question1_part1():
    x, y = sp.symbols('x y')
    f_sym = x**2 + 100 * y**2
    f = sp.lambdify(args=[x, y], expr=f_sym)

    iters = 200
    initial_values = [2, 2]

    optimizers = [
        ('Polyak', polyak_step_size(f_sym, [x, y], initial_values, iters=iters)),
        (
            'RMSProp (α=0.2, β=0.9)',
            rms_prop(
                f_sym,
                [x, y],
                0.2,
                0.9,
                initial_values,
                iters,
            ),
        ),
        (
            'Heavy Ball (α=0.01, β=0.9)',
            heavy_ball(
                f_sym,
                [x, y],
                0.01,
                0.9,
                initial_values,
                iters,
            ),
        ),
        (
            'Adam (α=0.1, β_1=0.9, β_2=0.999)',
            adam(
                f_sym,
                [x, y],
                0.1,
                0.9,
                0.999,
                initial_values,
                iters,
            ),
        ),
    ]

    for label, (_, steps) in optimizers:
        steps = np.array(steps)
        plt.plot(range(len(steps)), f(steps[:, 0], steps[:, 1]), label=label)

    plt.title(f'Gradient Descent on f(x,y)={f_sym}')
    plt.xlabel('Iterations')
    plt.ylabel('Function Value')
    plt.yscale('log')
    plt.grid(visible=True)
    plt.legend()
    plt.savefig('./images/question1.I.png', dpi=300, bbox_inches='tight')
    plt.show()


def question1_part2():
    x, y = sp.symbols('x y')
    f_sym = x**2 + 100 * y**2

    iters = 200
    intitial_values = [2, 2]
    alphas = [0.006, 0.01, 0.02]
    for alpha in alphas:
        _, heavy_ball_steps = heavy_ball(
            f_sym, [x, y], alpha, beta=0.9, init_vals=intitial_values, iters=iters
        )
        heavy_ball_steps = np.array(heavy_ball_steps)

        xs = heavy_ball_steps[:, 0]
        plt.plot(range(len(xs)), xs, label=f'{alpha=}')

    plt.title(f'Heavy Ball on f(x,y)={f_sym}')
    plt.xlabel('Iterations')
    plt.ylabel('x-values')

    plt.yscale('symlog')

    plt.grid(visible=True)
    plt.legend()
    plt.savefig('./images/question1.II.png', dpi=300, bbox_inches='tight')
    plt.show()


def question1_part3():
    x, y = sp.symbols('x y')
    f_sym = x**2 + 100 * y**2
    f = sp.lambdify(args=[x, y], expr=f_sym)

    iters = 200
    init_vals = [2, 2]

    X, Y = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))

    plt.contour(X, Y, f(X, Y), levels=10)

    _, hb_steps = heavy_ball(
        f_sym, [x, y], alpha=0.01, beta=0.9, init_vals=init_vals, iters=iters
    )

    plt.plot(*zip(*hb_steps), label='heavy ball')

    _, adam_steps = adam(
        f_sym,
        [x, y],
        alpha=0.1,
        beta_1=0.9,
        beta_2=0.999,
        init_vals=init_vals,
        iters=iters,
    )

    plt.plot(*zip(*adam_steps), label='Adam (α=0.1, β_1=0.9, β_2=0.999)')

    _, rms_steps = rms_prop(
        f_sym, [x, y], alpha=0.02, beta=0.9, init_vals=init_vals, iters=iters
    )

    plt.plot(*zip(*rms_steps), label='RMSProp (α=0.2, β=0.9)')

    plt.title(f'Gradient Descent on f(x,y)={f_sym}')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.legend()
    plt.savefig('./images/question1.III.png', dpi=300, bbox_inches='tight')
    plt.show()


def question2_part1():
    x, y = sp.symbols('x y')
    f_sym = (1 - x) ** 2 + 100 * (y - x**2) ** 2
    f = sp.lambdify(args=[x, y], expr=f_sym)

    iters = 3000
    init_vals = [-1.25, 2]

    optimizers = [
        ('Polyak', polyak_step_size(f_sym, [x, y], init_vals, iters=iters)),
        (
            'RMSProp (α=0.2, β=0.9)',
            rms_prop(
                f_sym,
                [x, y],
                0.2,
                0.9,
                init_vals,
                iters,
            ),
        ),
        (
            'Heavy Ball (α=0.0002, β=0.9)',
            heavy_ball(
                f_sym,
                [x, y],
                0.0002,
                0.9,
                init_vals,
                iters,
            ),
        ),
        (
            'Adam (α=0.05, β_1=0.9, β_2=0.999)',
            adam(
                f_sym,
                [x, y],
                0.05,
                0.9,
                0.999,
                init_vals,
                iters,
            ),
        ),
    ]

    for label, (_, steps) in optimizers:
        steps = np.array(steps)
        plt.plot(range(len(steps)), f(steps[:, 0], steps[:, 1]), label=label)

    plt.title(f'Gradient Descent on f(x,y)={f_sym}')
    plt.xlabel('Iterations')
    plt.ylabel('Function Value')
    plt.yscale('log')
    plt.grid(visible=True)
    plt.legend()
    plt.savefig('./images/question2.I.png', dpi=300, bbox_inches='tight')
    plt.show()


def question2_part2():
    x, y = sp.symbols('x y')
    f_sym = (1 - x) ** 2 + 100 * (y - x**2) ** 2

    iters = 3000
    intitial_values = [-1.25, 0.5]
    alphas = [0.02, 0.05, 0.12]
    for alpha in alphas:
        _, steps = adam(
            f_sym,
            [x, y],
            alpha,
            beta_1=0.9,
            beta_2=0.999,
            init_vals=intitial_values,
            iters=iters,
        )
        steps = np.array(steps)

        xs = steps[:, 0]
        plt.plot(range(len(xs)), xs, label=f'{alpha=}')

    plt.title(f'Adam on f(x,y)={f_sym}')
    plt.xlabel('Iterations')
    plt.ylabel('x-values')

    plt.grid(visible=True)
    plt.legend()
    plt.savefig('./images/question2.II.png', dpi=300, bbox_inches='tight')
    plt.show()


def question2_part3():
    x, y = sp.symbols('x y')
    f_sym = (1 - x) ** 2 + 100 * (y - x**2) ** 2
    f = sp.lambdify(args=[x, y], expr=f_sym)

    iters = 3000
    init_vals = [-1.25, 0.5]

    space = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(space, space)

    plt.contour(X, Y, f(X, Y), levels=20)

    _, hb_steps = heavy_ball(
        f_sym, [x, y], alpha=0.0002, beta=0.9, init_vals=init_vals, iters=iters
    )

    plt.plot(*zip(*hb_steps), label='heavy ball')

    _, adam_steps = adam(
        f_sym,
        [x, y],
        0.05,
        0.9,
        0.999,
        init_vals,
        iters,
    )

    plt.plot(*zip(*adam_steps), label='Adam (α=0.05, β_1=0.9, β_2=0.999)')

    _, rms_steps = rms_prop(
        f_sym, [x, y], alpha=0.01, beta=0.9, init_vals=init_vals, iters=iters
    )

    plt.plot(*zip(*rms_steps), label='RMSProp (α=0.0.1, β=0.9)')

    plt.title(f'Gradient Descent on f(x,y)={f_sym}')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.legend()
    plt.savefig('./images/question2.III.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    if not os.path.exists('./images'):
        os.makedirs('./images')

    question1_part1()
    question1_part2()
    question1_part3()

    question2_part1()
    question2_part2()
    question2_part3()
