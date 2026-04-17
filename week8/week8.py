import os
from typing import Callable

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Rectangle

sp.init_printing(use_unicode=True, use_latex=False)
plt.rcParams.update(
    {
        "figure.figsize": (12, 6),
        "axes.grid": False,
        "grid.alpha": 0.3,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "lines.linewidth": 1,
        "lines.markersize": 3,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def project_X1(z):
    x1 = np.clip(z[0], 0.5, 2.5)
    x2 = np.clip(z[1], 0.5, 3.5)
    return np.array([x1, x2])


def project_X2(z):
    x1 = np.clip(z[0], 0.5, None)
    x2 = np.clip(z[1], 0.5, None)
    if x1 <= x2:
        return np.array([x1, x2])

    mid = (x1 + x2) / 2
    return np.array([mid, mid])


def project_Q3(z):
    x1 = np.clip(z[0], 0.5, 3.0)
    x2 = np.clip(z[1], 0.5, 3.0)
    if x1 + x2 > 4:
        correction = (x1 + x2 - 4) / 2
        x1 = x1 - correction
        x2 = x2 - correction
    return np.array([x1, x2])


def dual_primal_gd(
    fn,
    args,
    alpha,
    beta,
    init_vals,
    iters=50,
):
    xs = np.array(init_vals)
    xs_steps = [init_vals]

    derivatives = [sp.diff(fn, arg) for arg in args]

    derivatives = [sp.lambdify(args, expr=d) for d in derivatives]
    for _ in range(iters):
        grads = np.array([d(*xs) for d in derivatives])

        xs = xs - alpha * grads
        xs_steps.append(xs.tolist())

    return xs, xs_steps


def gradient_descent(
    fn,
    args,
    alpha,
    init_vals,
    iters=50,
):
    xs = np.array(init_vals)
    xs_steps = [init_vals]

    derivatives = [sp.diff(fn, arg) for arg in args]

    derivatives = [sp.lambdify(args, expr=d) for d in derivatives]
    for _ in range(iters):
        grads = np.array([d(*xs) for d in derivatives])

        xs = xs - alpha * grads
        xs_steps.append(xs.tolist())

    return xs, xs_steps


def projected_gradient_descent(
    fn,
    args,
    alpha,
    init_vals,
    projection: Callable,
    iters=50,
):
    xs = np.array(init_vals)
    xs_steps = [init_vals]

    derivatives = [sp.diff(fn, arg) for arg in args]

    derivatives = [sp.lambdify(args, expr=d) for d in derivatives]
    for _ in range(iters):
        grads = np.array([d(*xs) for d in derivatives])

        zs = xs - alpha * grads
        xs = projection(zs)
        xs_steps.append(xs.tolist())

    return xs, xs_steps


def question_1_X1():
    x1, x2 = sp.symbols("x1 x2")
    func = (x1 - 1.2) ** 2 + 2 * (x2 - 2.5) ** 2 + 0.4 * x1 * x2
    dfdx1 = sp.diff(func, x1)
    dfdx2 = sp.diff(func, x2)
    print(dfdx1, dfdx2)

    _, xs_steps = projected_gradient_descent(
        fn=func,
        args=[x1, x2],
        alpha=0.05,
        init_vals=[2.4, 0.7],
        projection=project_X1,
        iters=60,
    )

    f = sp.lambdify((x1, x2), func, "numpy")

    losses = [f(*xs) for xs in xs_steps]

    space_x1 = np.linspace(0.2, 2.8, 200)
    space_x2 = np.linspace(0.2, 3.8, 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = f(X1, X2)

    _, ax1 = plt.subplots()
    ax1.set_title("Question 1X1: Contour and Projected GD Trajectory")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    ax1.contour(X1, X2, Z, levels=30, cmap="plasma")

    feasible_x1_min, feasible_x1_max = 0.5, 2.5
    feasible_x2_min, feasible_x2_max = 0.5, 3.5
    width = feasible_x1_max - feasible_x1_min
    height = feasible_x2_max - feasible_x2_min

    rect = Rectangle(
        (feasible_x1_min, feasible_x2_min),
        width,
        height,
        facecolor="blue",
        alpha=0.25,
        edgecolor="black",
        linestyle="--",
        linewidth=2,
        zorder=10,
        label="Feasible Region",
    )
    ax1.add_patch(rect)

    traj = np.array(xs_steps)
    ax1.plot(traj[:, 0], traj[:, 1], marker="o", color="blue", label="Trajectory")
    ax1.scatter(
        *traj[-1],
        c="red",
        zorder=5,
        label=f"Last ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})",
    )
    ax1.legend()

    plt.savefig("./week8/images/question1_X1_contour.png")
    plt.show()

    _, ax2 = plt.subplots()
    ax2.set_title("Loss vs. Iterations (log scale)")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("f(x) (log)")
    ax2.plot(range(len(losses)), losses, color="blue", marker="o", label="Loss")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.savefig("./week8/images/question1_X1_loss.png")
    plt.show()

    _, ax3 = plt.subplots()
    ax3.set_title("x1 and x2 vs. Iterations")
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Value")
    ax3.plot(range(len(traj)), traj[:, 0], marker="o", color="red", label="x1")
    ax3.plot(range(len(traj)), traj[:, 1], marker="s", color="green", label="x2")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.savefig("./week8/images/question1_X1_variables.png")
    plt.show()


def question_1_X2():
    x1, x2 = sp.symbols("x1 x2")
    func = (x1 - 1.2) ** 2 + 2 * (x2 - 2.5) ** 2 + 0.4 * x1 * x2
    dfdx1 = sp.diff(func, x1)
    dfdx2 = sp.diff(func, x2)
    print(dfdx1, dfdx2)

    _, xs_steps = projected_gradient_descent(
        fn=func,
        args=[x1, x2],
        alpha=0.05,
        init_vals=[2.4, 0.7],
        projection=project_X2,
        iters=60,
    )

    f = sp.lambdify((x1, x2), func, "numpy")

    losses = [f(*xs) for xs in xs_steps]

    space_x1 = np.linspace(0.2, 2.8, 200)
    space_x2 = np.linspace(0.2, 3.8, 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = f(X1, X2)

    _, ax1 = plt.subplots()
    ax1.set_title("Question 1 X2: Contour and Projected GD Trajectory")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    ax1.contour(X1, X2, Z, levels=30, cmap="plasma")

    feasible_vertices = [(0.5, 0.5), (2.8, 2.8), (2.8, 3.8), (0.5, 3.8)]
    feasible_patch = Polygon(
        feasible_vertices,
        facecolor="blue",
        alpha=0.25,
        edgecolor="black",
        linestyle="--",
        linewidth=2,
        zorder=10,
        label="Feasible Region",
    )
    ax1.add_patch(feasible_patch)

    traj = np.array(xs_steps)
    ax1.plot(traj[:, 0], traj[:, 1], marker="o", color="blue", label="Trajectory")
    ax1.scatter(
        *traj[-1],
        c="red",
        zorder=5,
        label=f"Last ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})",
    )
    ax1.legend()

    plt.savefig("./week8/images/question1_X2_contour.png")
    plt.show()

    _, ax2 = plt.subplots()
    ax2.set_title("Loss vs. Iterations (log scale)")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("f(x) (log)")
    ax2.plot(range(len(losses)), losses, color="blue", marker="o", label="Loss")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.savefig("./week8/images/question1_X2_loss.png")
    plt.show()

    _, ax3 = plt.subplots()
    ax3.set_title("x1 and x2 vs. Iterations")
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Value")
    ax3.plot(range(len(traj)), traj[:, 0], marker="o", color="red", label="x1")
    ax3.plot(range(len(traj)), traj[:, 1], marker="s", color="green", label="x2")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.savefig("./week8/images/question1_X2_variables.png")
    plt.show()


def question_2_fixed_penalty():
    x1, x2 = sp.symbols("x1 x2")

    # lam1, lam2 = 0.5, 0.5
    # alpha = 0.05

    lam1, lam2 = 4, 4
    alpha = 0.025

    init_vals = [1.4, 0.6]

    f = (x1 - 0.2) ** 2 + (x2 - 2) ** 2

    g1 = 0.5 - x1
    g2 = 1 - x1 * x2

    Q1 = sp.Piecewise((g1, g1 > 0), (0, True))
    Q2 = sp.Piecewise((g2, g2 > 0), (0, True))

    Q = lam1 * Q1 + lam2 * Q2

    F = f + Q
    xs = np.array(init_vals)
    steps = [init_vals]

    derivatives = [sp.diff(F, arg) for arg in [x1, x2]]
    derivatives = [sp.lambdify([x1, x2], expr=d) for d in derivatives]

    for _ in range(60):
        grads = np.array([d(*xs) for d in derivatives])

        xs = xs - alpha * grads
        steps.append(xs.tolist())

    f_lambda = sp.lambdify([x1, x2], f)

    space_x1 = np.linspace(0.1, 1.5, 200)
    space_x2 = np.linspace(0.2, 3, 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = f_lambda(X1, X2)

    plt.title(f"Contour of {f} and GD Trajectory Lambda={lam1}")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.contour(X1, X2, Z, levels=30, cmap="plasma")

    x1_vals = np.linspace(0.5, 1.5, 300)

    x2_hyperbola = 1 / x1_vals

    x2_upper = 3

    plt.fill_between(
        x1_vals,
        x2_hyperbola,
        x2_upper,
        alpha=0.25,
        color="blue",
        label="Feasible Region",
    )

    plt.plot(x1_vals, x2_hyperbola, color="black", linestyle="--", linewidth=1.5)
    plt.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5)
    traj = np.array(steps)
    plt.plot(traj[:, 0], traj[:, 1], marker="o", color="blue", label="Trajectory")
    plt.scatter(
        *traj[-1],
        c="red",
        zorder=5,
        label=f"Last ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})",
    )
    plt.scatter(x=0.2, y=2, c="blue", label="true optimum")
    plt.legend()

    plt.savefig(f"./week8/images/question2_contour_lambda_{lam1}.png")
    plt.show()

    steps = np.array(steps)

    g1_vals = 0.5 - steps[:, 0]
    g2_vals = 1 - steps[:, 0] * steps[:, 1]

    plt.plot(range(len(g1_vals)), g1_vals, label="g1 (x1 >= 0.5)")
    plt.plot(range(len(g2_vals)), g2_vals, label="g2 (x1*x2 >= 1)")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Iterations")
    plt.ylabel("Constraint violation")
    plt.title(f"Constraint violations vs iterations Lambda={lam1}")
    plt.legend()
    plt.savefig(f"./week8/images/question2_violation_lambda_{lam1}.png")
    plt.show()


def question_2_primal_dual():
    initial = [1.4, 0.6]
    x1, x2 = sp.symbols("x1 x2")
    lam1, lam2 = 0.0, 0.0
    alpha = 0.06
    beta = 0.08

    f = (x1 - 0.2) ** 2 + (x2 - 2) ** 2

    g1 = 0.5 - x1
    g2 = 1 - x1 * x2

    xs = np.array(initial)
    xs_steps = [xs]
    lam_steps = [(lam1, lam2)]

    dfdx1 = sp.diff(f, x1)
    dfdx2 = sp.diff(f, x2)

    dfdx1 = sp.lambdify([x1, x2], dfdx1)
    dfdx2 = sp.lambdify([x1, x2], dfdx2)

    dg1dx1 = sp.diff(g1, x1)
    dg1dx2 = sp.diff(g1, x2)
    dg1dx1 = sp.lambdify([x1, x2], dg1dx1)
    dg1dx2 = sp.lambdify([x1, x2], dg1dx2)

    dg2dx1 = sp.diff(g2, x1)
    dg2dx2 = sp.diff(g2, x2)
    dg2dx1 = sp.lambdify([x1, x2], dg2dx1)
    dg2dx2 = sp.lambdify([x1, x2], dg2dx2)

    g1 = sp.lambdify([x1, x2], g1)
    g2 = sp.lambdify([x1, x2], g2)

    for _ in range(200):
        grads = np.array(
            [
                dfdx1(*xs) + lam1 * dg1dx1(*xs) + lam2 * dg2dx1(*xs),
                dfdx2(*xs) + lam1 * dg1dx2(*xs) + lam2 * dg2dx2(*xs),
            ]
        )

        xs = xs - alpha * grads
        lam1 = max(0, lam1 + beta * g1(*xs))
        lam2 = max(0, lam2 + beta * g2(*xs))

        xs_steps.append(xs.tolist())
        lam_steps.append((lam1, lam2))

    f_lambda = sp.lambdify([x1, x2], f)

    space_x1 = np.linspace(0.1, 1.5, 200)
    space_x2 = np.linspace(0.2, 3, 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = f_lambda(X1, X2)

    _, ax1 = plt.subplots()
    ax1.set_title("Question 2 Primary-Dual: Contour and Trajectory")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    ax1.contour(X1, X2, Z, levels=30, cmap="plasma")

    x1_vals = np.linspace(0.5, 1.5, 300)
    x2_hyperbola = 1 / x1_vals
    x2_upper = 3

    ax1.fill_between(
        x1_vals,
        x2_hyperbola,
        x2_upper,
        alpha=0.25,
        color="blue",
        label="Feasible Region",
    )

    ax1.plot(x1_vals, x2_hyperbola, color="black", linestyle="--", linewidth=1.5)
    ax1.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5)

    traj = np.array(xs_steps)
    ax1.plot(traj[:, 0], traj[:, 1], marker="o", color="blue", label="Trajectory")
    ax1.scatter(
        *traj[-1],
        c="red",
        zorder=5,
        label=f"Last ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})",
    )
    ax1.scatter(x=0.2, y=2, c="blue", label="True optimum")
    ax1.legend()

    plt.savefig("./week8/images/question2_primary_dual_contour.png")
    plt.show()

    _, ax2 = plt.subplots()
    ax2.set_title("x1 and x2 vs. Iterations")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Value")
    ax2.plot(range(len(traj)), traj[:, 0], marker="o", color="red", label="x1")
    ax2.plot(range(len(traj)), traj[:, 1], marker="s", color="green", label="x2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.savefig("./week8/images/question2_primary_dual_variables.png")
    plt.show()

    lam_trajs = np.array(lam_steps)
    _, ax3 = plt.subplots()
    ax3.set_title("λ1 and λ2 vs. Iterations")
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Value")
    ax3.plot(
        range(len(lam_trajs)), lam_trajs[:, 0], marker="o", color="red", label="λ1"
    )
    ax3.plot(
        range(len(lam_trajs)), lam_trajs[:, 1], marker="s", color="green", label="λ2"
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.savefig("./week8/images/question2_primary_dual_lambdas.png")
    plt.show()


def question_3_frank_wolfe():
    x1, x2 = sp.symbols("x1 x2")
    f = (x1 - 1.5) ** 2 + (x2 - 1.2) ** 2

    dfdx1 = sp.diff(f, x1)
    dfdx2 = sp.diff(f, x2)

    dfdx1 = sp.lambdify([x1, x2], dfdx1)
    dfdx2 = sp.lambdify([x1, x2], dfdx2)

    vertices = np.array([[0.5, 0.5], [3, 0.5], [3, 1], [1, 3], [0.5, 3]])
    # beta = 0.95
    beta = 0.8

    xs = np.array([2.8, 0.8])
    steps = [xs]
    z_steps = []

    for _ in range(50):
        grads = np.array([dfdx1(*xs), dfdx2(*xs)])
        scores = vertices @ grads
        z = vertices[np.argmin(scores)]
        z_steps.append(z.copy())
        xs = beta * xs + (1 - beta) * z
        steps.append(xs.copy())

    f_lambda = sp.lambdify([x1, x2], f)

    losses = [f_lambda(*xs) for xs in steps]

    space_x1 = np.linspace(0.3, 3.2, 200)
    space_x2 = np.linspace(0.3, 3.2, 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = f_lambda(X1, X2)

    _, ax1 = plt.subplots()
    ax1.set_title(f"Frank-Wolfe Contour and Trajectory Beta={beta}")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    ax1.contour(X1, X2, Z, levels=30, cmap="plasma")

    feasible_patch = Polygon(
        vertices,
        facecolor="blue",
        alpha=0.25,
        edgecolor="black",
        linestyle="--",
        linewidth=2,
        zorder=10,
        label="Feasible Region",
    )
    ax1.add_patch(feasible_patch)

    ax1.scatter(
        vertices[:, 0],
        vertices[:, 1],
        c="black",
        zorder=15,
        s=50,
        marker="o",
        label="Vertices",
    )

    traj = np.array(steps)
    ax1.plot(traj[:, 0], traj[:, 1], marker="o", color="blue", label="Trajectory")
    ax1.scatter(
        *traj[-1],
        c="red",
        zorder=5,
        label=f"Last ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})",
    )
    ax1.scatter(x=1.5, y=1.2, c="blue", label="True optimum (1.5, 1.2)")
    ax1.legend()

    plt.savefig(f"./week8/images/question3_contour_beta_{beta}.png")
    plt.show()

    _, ax2 = plt.subplots()
    ax2.set_title("f(x) vs. Iterations")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("f(x)")
    ax2.plot(range(len(losses)), losses, marker="o", color="blue", label="Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.savefig(f"./week8/images/question3_loss_beta_{beta}.png")
    plt.show()

    _, ax3 = plt.subplots()
    ax3.set_title("x_t and z_t vs. Iterations")
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Value")
    traj = np.array(steps)
    z_traj = np.array(z_steps)
    ax3.plot(range(len(traj)), traj[:, 0], marker="o", color="red", label="x1")
    ax3.plot(range(len(traj)), traj[:, 1], marker="s", color="green", label="x2")
    ax3.scatter(
        range(len(z_traj)), z_traj[:, 0], marker="o", s=20, color="blue", label="z1"
    )
    ax3.scatter(
        range(len(z_traj)), z_traj[:, 1], marker="o", color="orange", s=10, label="z2"
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.savefig(f"./week8/images/question3_xt_zt_beta_{beta}.png")
    plt.show()


def question_3_projected():
    x1, x2 = sp.symbols("x1 x2")
    f = (x1 - 1.5) ** 2 + (x2 - 1.2) ** 2

    _, steps = projected_gradient_descent(
        fn=f,
        args=[x1, x2],
        alpha=0.05,
        init_vals=[2.8, 0.8],
        projection=project_Q3,
        iters=50,
    )

    f_lambda = sp.lambdify([x1, x2], f)

    vertices = np.array([[0.5, 0.5], [3, 0.5], [3, 1], [1, 3], [0.5, 3]])

    space_x1 = np.linspace(0.3, 3.2, 200)
    space_x2 = np.linspace(0.3, 3.2, 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = f_lambda(X1, X2)

    _, ax1 = plt.subplots()
    ax1.set_title(f"PGD Contour and Trajectory")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    ax1.contour(X1, X2, Z, levels=30, cmap="plasma")

    feasible_patch = Polygon(
        vertices,
        facecolor="blue",
        alpha=0.25,
        edgecolor="black",
        linestyle="--",
        linewidth=2,
        zorder=10,
        label="Feasible Region",
    )

    ax1.add_patch(feasible_patch)

    traj = np.array(steps)
    ax1.plot(traj[:, 0], traj[:, 1], marker="o", color="blue", label="Trajectory")
    ax1.scatter(
        *traj[-1],
        c="red",
        zorder=5,
        label=f"Last ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})",
    )
    ax1.scatter(x=1.5, y=1.2, c="blue", label="True optimum (1.5, 1.2)")
    ax1.legend()

    plt.savefig(f"./week8/images/question3_pgd_contour.png")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists("./week8/images"):
        os.makedirs("./week8/images")
    question_1_X1()
    question_1_X2()
    question_2_fixed_penalty()
    question_2_primal_dual()
    question_3_frank_wolfe()
    question_3_projected()
