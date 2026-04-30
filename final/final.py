import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.linalg import inv

plt.rcParams["savefig.dpi"] = 150
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["image.cmap"] = "plasma"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["legend.fontsize"] = 8


def setup_benchmark_A(noise_scaler=1):
    X = np.random.normal(0, 1, (1000, 2))
    theta_star = np.array([3, 4]).reshape(2, 1)
    eps = np.random.normal(0, 1 * noise_scaler, (1000, 1))

    Y = X @ theta_star + eps

    def mse(X, theta, y):
        m = len(X)
        error = X @ theta - y
        return (1 / (2 * m)) * np.sum(error**2)

    def grad_fn(X, theta, y):
        m = len(X)
        return (1 / m) * X.T @ (X @ theta - y)

    def hessian_fn(_):
        m = len(X)
        return (1 / m) * X.T @ X

    return X, Y, mse, grad_fn, hessian_fn


def setup_benchmark_B():

    def net(x):
        x1, x2 = x[0, 0], x[1, 0]
        return ((x1 - 1) ** 2) + (5 * (x2 - 2) ** 2) + np.sin(x1)

    def grad_fn(x):
        x1, x2 = x[0, 0], x[1, 0]
        dfdx1 = 2 * (x1 - 1) + np.cos(x1)
        dfdx2 = 10 * (x2 - 2)
        return np.array([dfdx1, dfdx2]).reshape(-1, 1)

    def finite_diff(x, delta):
        fx = net(x)
        e1 = np.array([[delta], [0]])
        e2 = np.array([[0], [delta]])

        dfdx1 = (net(x + e1) - fx) / delta
        dfdx2 = (net(x + e2) - fx) / delta
        return np.array([dfdx1, dfdx2]).reshape(-1, 1)

    def nesterov_random_dir(x, delta):
        u = np.random.randn(2, 1)
        u = u / np.linalg.norm(u)

        fx = net(x)

        return ((net(x + delta * u) - fx) / delta) * u

    def hessian_fn(x):
        x1 = x[0, 0]
        d2fdx1dx1 = 2 - np.sin(x1)
        d2fdx1dx2 = 0
        d2fdx2dx2 = 10
        d2fdx2dx1 = 0
        return np.array([[d2fdx1dx1, d2fdx1dx2], [d2fdx2dx1, d2fdx2dx2]])

    return net, grad_fn, hessian_fn, finite_diff, nesterov_random_dir


def setup_benchmark_C():
    def rosenbrock(x):
        x1, x2 = x[0, 0], x[1, 0]
        return ((1 - x1) ** 2) + (100 * (x2 - x1**2) ** 2)

    def grad_fn(x):
        x1, x2 = x1, x2 = x[0, 0], x[1, 0]
        dfdx1 = (-2 * (1 - x1)) - 400 * x1 * (x2 - x1**2)
        dfdx2 = 200 * (x2 - x1**2)
        return np.array([dfdx1, dfdx2]).reshape(-1, 1)

    def hessian_fn(x):
        x1, x2 = x1, x2 = x[0, 0], x[1, 0]
        d2fdx1dx1 = 2 + 1200 * x1**2 - 400 * x2
        d2fdx1dx2 = -400 * x1
        d2fdx2dx1 = -400 * x1
        d2fdx2dx2 = 200
        return np.array([[d2fdx1dx1, d2fdx1dx2], [d2fdx2dx1, d2fdx2dx2]])

    return rosenbrock, grad_fn, hessian_fn


def polyak(loss_fn, grad_fn, x0, f_star, eps, iters):
    xi = np.array([x0]).reshape(-1, 1)
    steps = [xi.copy()]
    alphas = []

    for _ in range(iters):
        grads = grad_fn(xi)

        alpha = (loss_fn(xi) - f_star) / (np.sum(grads**2) + eps)
        xi = xi - alpha * grads

        steps.append(xi.copy())
        alphas.append(alpha)

    return steps, alphas


def adagrad(grad_fn, x0, alpha0, iters):
    xi = np.array([x0]).reshape(-1, 1)
    alpha_t = alpha0
    accumulator = np.zeros_like(xi, dtype=np.dtypes.Float64DType)

    def step_fn(x):
        nonlocal alpha_t, accumulator
        grads = grad_fn(x)

        accumulator += grads**2

        alpha_t = alpha0 / np.sqrt(accumulator + 10**-5)

        return alpha_t * grads, alpha_t

    steps = [xi.copy()]
    alphas = []
    for _ in range(iters):
        step, alpha_t = step_fn(xi)
        xi = xi - step

        steps.append(xi.copy())
        alphas.append(alpha_t)

    return steps, alphas


def rms_prop(grad_fn, x0, alpha, beta, iters):
    a0 = alpha
    xi = np.array([x0]).reshape(-1, 1)
    steps = [xi]
    alphas = []

    accumulator = 0
    for _ in range(iters):
        grads = grad_fn(xi)
        accumulator = beta * accumulator + (1 - beta) * grads**2

        alpha = a0 / (np.sqrt(accumulator) + 10**-5)
        xi = xi - alpha * grads

        steps.append(xi.copy())
        alphas.append(alpha)

    return steps, alphas


def heavy_ball(grad_fn, alpha, beta, x0, iters):
    xi = np.array(x0).reshape(-1, 1)
    steps = [xi.copy()]
    alphas = []

    z = np.zeros_like(xi)
    for _ in range(iters):
        grads = grad_fn(xi)
        z = beta * z + alpha * grads
        xi = xi - z

        steps.append(xi.copy())
        alphas.append(z)

    return steps, alphas


def gradient_descent(grad_fn, alpha, x0, iters):
    xi = np.array(x0, dtype=np.dtypes.Float64DType).reshape(-1, 1)
    steps = [xi.copy()]
    magnitudes = []

    for _ in range(iters):
        grads = grad_fn(xi)
        step = alpha * grads
        xi -= step

        steps.append(xi.copy())
        magnitudes.append(np.linalg.norm(step))

    return steps, magnitudes


def nesterov_momentum(grad_fn, x0, alpha, beta_max, iters):
    xi = np.array([x0]).reshape(-1, 1)

    z = 0

    def step_fn(x, k):
        nonlocal z
        beta = min((k - 1) / (k + 2), beta_max)

        z = beta * z - alpha * grad_fn(x + beta * z)
        return -z

    steps = [xi.copy()]

    for k in range(1, iters + 1):
        step = step_fn(xi, k)
        xi = xi - step

        steps.append(xi.copy())

    return steps


def adam(grad_fn, x0, alpha, beta_1, beta_2, iters):
    xi = np.array([x0]).reshape(-1, 1)
    steps = [xi.copy()]

    v = np.zeros_like(xi)
    m = np.zeros_like(xi)
    for t in range(1, iters + 1):
        grads = grad_fn(xi)

        m = beta_1 * m + (1 - beta_1) * grads
        v = beta_2 * v + (1 - beta_2) * grads**2

        m_hat = m / (1 - beta_1**t)
        v_hat = v / (1 - beta_2**t)

        xi = xi - alpha * m_hat / (np.sqrt(v_hat) + 10**-8)
        steps.append(xi.copy())

    return steps


def stochastic_gradient_descent(
    loss_fn,
    grad_fn,
    X,
    y,
    x0,
    alpha,
    max_epochs,
    batch_size,
):
    rng = np.random.default_rng(67)

    xi = np.array(x0).reshape(-1, 1)
    steps = [xi.copy()]
    losses = [loss_fn(X, xi, y).item()]

    epoch = 0
    while epoch < max_epochs:
        indices = rng.permutation(len(X))
        X_shuff = X[indices]
        y_shuff = y[indices]

        for batch in range(0, len(X), batch_size):
            X_batch = X_shuff[batch : batch + batch_size]
            y_batch = y_shuff[batch : batch + batch_size]

            step = alpha * grad_fn(X_batch, xi, y_batch)
            xi = xi - step

            steps.append(xi.copy())
            losses.append(loss_fn(X, xi, y).item())
        epoch += 1

    return steps, losses


def newton_update(grad_fn, x0, hessian_fn, alpha, iters):
    xi = np.array(x0).reshape(-1, 1)
    steps = [xi.copy()]
    magnitudes = []
    I = np.eye(2, 2)
    damping = 10**-8
    for _ in range(iters):
        step = alpha * inv(hessian_fn(xi) + damping * I) @ grad_fn(xi)
        xi = xi - step

        steps.append(xi.copy())
        magnitudes.append(np.linalg.norm(step))

    return steps, magnitudes


def nelder_mead(fn, x0, simplex_step, iters):
    n = len(x0)
    simplex = [np.array(x0).reshape(-1, 1)]
    for i in range(n):
        e_i = np.zeros((n, 1))
        e_i[i, 0] = 1
        simplex.append(simplex[0] + simplex_step * e_i)

    steps = [simplex[0].copy()]
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    for _ in range(iters):
        values = np.array([fn(p).item() for p in simplex])
        ordering = np.argsort(values)
        simplex = [simplex[i] for i in ordering]
        centroid = sum(simplex[:-1]) / n
        worst = simplex[-1]
        reflected = centroid + alpha * (centroid - worst)

        f_r = fn(reflected).item()
        f_best = values[0]
        f_worst = values[-1]
        f_second_worst = values[-2]

        if f_r < f_best:
            expanded = centroid + gamma * (reflected - centroid)
            simplex[-1] = expanded if fn(expanded).item() < f_r else reflected
        elif f_r < f_second_worst:
            simplex[-1] = reflected
        else:
            contracted = centroid + rho * (worst - centroid)
            if fn(contracted).item() < f_worst:
                simplex[-1] = contracted
            else:
                for j in range(1, n + 1):
                    simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
        steps.append(simplex[0].copy())

    return steps


def grid_search(loss_fn):
    steps = []
    fx_samples = []

    x1_space = np.linspace(-2, 2, 55)
    x2_space = np.linspace(-1, 3, 55)

    for x1 in x1_space:
        for x2 in x2_space:
            x = np.array([x1, x2]).reshape(-1, 1)
            fx = loss_fn(x)
            steps.append((x1, x2))
            fx_samples.append((fx, [x1, x2]))

    return steps, fx_samples


def projected_gradient_descent(
    grad_fn,
    alpha,
    x0,
    iters,
):
    xi = np.array(x0).reshape(-1, 1)

    def projection(x):
        x1, x2 = x[0, 0], x[1, 0]
        x1 = np.clip(x1, 0.5, None)
        return np.array([[x1], [x2]])

    steps = [xi.copy()]

    for _ in range(iters):
        zs = xi - alpha * grad_fn(xi)
        xi = projection(zs)
        steps.append(xi.copy())

    return steps


def plot_loss(losses, title, ylabel, savepath):
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel(ylabel)
    ax.plot(range(len(losses)), losses, color="blue", label="Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    plt.savefig(savepath)
    plt.show()


def plot_scalar_alpha(alphas, title, savepath):
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Alpha")
    ax.plot(range(len(alphas)), alphas, color="blue", marker="o", label="Alpha")
    ax.legend()
    ax.grid(True)
    plt.savefig(savepath)
    plt.show()


def plot_contour_net(steps, title, savepath):

    traj = np.array([np.array(s).flatten() for s in steps])
    x1, x2 = traj[:, 0], traj[:, 1]
    x1_min, x1_max = np.min(x1) * 1.2, np.max(x1) * 1.2
    x2_min, x2_max = np.min(x2) * 1.2, np.max(x2) * 1.2

    x1_lower = min(x1_min, -2)
    x1_upper = max(x1_max, 4)

    x2_lower = min(x2_min, -1)
    x2_upper = max(x2_max, 5)

    space_x1 = np.linspace(x1_lower, x1_upper, 200)
    space_x2 = np.linspace(x2_lower, x2_upper, 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = (X1 - 1) ** 2 + 5 * (X2 - 2) ** 2 + np.sin(X1)
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.contour(X1, X2, Z, levels=30)
    ax.plot(traj[:, 0], traj[:, 1], marker="o", color="blue", label="Trajectory")
    ax.scatter(
        *traj[-1],
        c="red",
        zorder=5,
        label=f"Last ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})",
    )
    ax.legend()
    plt.savefig(savepath)
    plt.show()


def plot_contour_rosenbrock(steps, title, savepath):
    space_x1 = np.linspace(-1.5, 2.0, 200)
    space_x2 = np.linspace(-0.5, 2.5, 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = (1 - X1) ** 2 + 100 * (X2 - X1**2) ** 2

    traj = np.array([np.array(s).flatten() for s in steps])

    _, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.contour(X1, X2, Z, levels=30)
    ax.plot(traj[:, 0], traj[:, 1], marker="o", color="blue", label="Trajectory")
    ax.scatter(
        *traj[-1],
        c="red",
        zorder=5,
        label=f"Last ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})",
    )
    ax.scatter(1, 1, c="black", zorder=5, label="Global minimum (1, 1)")
    ax.legend()
    plt.savefig(savepath)
    plt.show()


BENCHMARK_LABELS = ["Linear Regression", "Toy Neural Net", "Rosenbrock"]
BENCHMARK_YLABELS = ["f(x) (MSE)", "f(x) Neural Net", "f(x) Rosenbrock"]


def plot_loss_subplots(losses_per_bench, optimizer_name, savepath, gd_losses_per_bench):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, losses, label, ylabel in zip(
        axes, losses_per_bench, BENCHMARK_LABELS, BENCHMARK_YLABELS
    ):
        ax.set_title(label)
        ax.set_xlabel("Iterations")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True)
    for ax, gd_losses in zip(axes, gd_losses_per_bench):
        ax.plot(range(len(gd_losses)), gd_losses, color="green", label="GD")
    for ax, losses in zip(axes, losses_per_bench):
        ax.plot(range(len(losses)), losses, color="blue", label=optimizer_name)
    for ax in axes:
        ax.legend()
    fig.suptitle(f"{optimizer_name}: Loss vs Iterations")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def plot_contour_subplots(
    steps_b2, steps_b3, optimizer_name, savepath, gd_steps_b2, gd_steps_b3
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if gd_steps_b2 is not None:
        gd_traj = np.array([np.array(s).flatten() for s in gd_steps_b2])
        ax1.plot(
            gd_traj[:, 0],
            gd_traj[:, 1],
            color="green",
            marker="o",
            ms=4,
            label=f"GD ({gd_traj[-1, 0]:.2f}, {gd_traj[-1, 1]:.2f})",
        )
    traj = np.array([np.array(s).flatten() for s in steps_b2])

    x1, x2 = traj[:, 0], traj[:, 1]
    x1_min, x1_max = np.min(x1) * 1.05, np.max(x1) * 1.05
    x2_min, x2_max = np.min(x2) * 1.05, np.max(x2) * 1.05

    x1_lower = min(x1_min, -2)
    x1_upper = max(x1_max, 4)

    x2_lower = min(x2_min, -1)
    x2_upper = max(x2_max, 5)

    space_x1 = np.linspace(x1_lower, x1_upper, 200)
    space_x2 = np.linspace(x2_lower, x2_upper, 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = (X1 - 1) ** 2 + 5 * (X2 - 2) ** 2 + np.sin(X1)
    ax1.contour(X1, X2, Z, levels=30)
    ax1.plot(
        traj[:, 0],
        traj[:, 1],
        color="blue",
        marker="o",
        ms=4,
        label=f"{optimizer_name} ({traj[-1, 0]:.2f}, {traj[-1, 1]:.2f})",
    )
    ax1.set_title("Toy Neural Net")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.legend()

    traj = np.array([np.array(s).flatten() for s in steps_b3])

    all_traj = traj
    if gd_steps_b3 is not None:
        gd_traj = np.array([np.array(s).flatten() for s in gd_steps_b3])
        all_traj = np.vstack([traj, gd_traj])

    x1, x2 = all_traj[:, 0], all_traj[:, 1]
    x1_min, x1_max = np.min(x1) * 1.05, np.max(x1) * 1.05
    x2_min, x2_max = np.min(x2) * 1.05, np.max(x2) * 1.05

    x1_lower = min(x1_min, -1.5)
    x1_upper = max(x1_max, 2.0)

    x2_lower = min(x2_min, -0.5)
    x2_upper = max(x2_max, 2.5)

    space_x1 = np.linspace(x1_lower, x1_upper, 200)
    space_x2 = np.linspace(x2_lower, x2_upper, 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = (1 - X1) ** 2 + 100 * (X2 - X1**2) ** 2
    ax2.contour(X1, X2, Z, levels=30)

    if gd_steps_b3 is not None:
        ax2.plot(
            gd_traj[:, 0],
            gd_traj[:, 1],
            color="green",
            marker="o",
            ms=4,
            label=f"GD ({gd_traj[-1, 0]:.2f}, {gd_traj[-1, 1]:.2f})",
        )
    ax2.plot(
        traj[:, 0],
        traj[:, 1],
        color="blue",
        marker="o",
        ms=4,
        label=f"{optimizer_name} ({traj[-1, 0]:.2f}, {traj[-1, 1]:.2f})",
    )
    ax2.scatter(1, 1, c="black", zorder=5, label="Global min (1,1)")
    ax2.set_title("Rosenbrock")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.legend()

    fig.suptitle(f"{optimizer_name}: Contour Trajectories")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def plot_scalar_alpha_subplots(alphas_per_bench, optimizer_name, savepath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, alphas, label in zip(axes, alphas_per_bench, BENCHMARK_LABELS):
        ax.set_title(label)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Alpha")
        ax.plot(range(len(alphas)), alphas, color="blue")
        ax.grid(True)
    fig.suptitle(f"{optimizer_name}: Alpha vs Iterations")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def plot_vector_alpha_subplots(
    alphas_per_bench,
    optimizer_name,
    savepath,
    param_names_per_bench=(("θ0", "θ1"), ("x1", "x2"), ("x1", "x2")),
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, alphas, label, param_names in zip(
        axes, alphas_per_bench, BENCHMARK_LABELS, param_names_per_bench
    ):
        alpha_arr = np.array([a.flatten() for a in alphas])
        ax.set_title(label)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Alpha")
        ax.plot(
            range(len(alpha_arr)),
            alpha_arr[:, 0],
            color="#1f77b4",
            label=f"α {param_names[0]}",
        )
        ax.plot(
            range(len(alpha_arr)),
            alpha_arr[:, 1],
            color="#d62728",
            label=f"α {param_names[1]}",
        )
        ax.legend()
        ax.grid(True)
    fig.suptitle(f"{optimizer_name}: Alpha vs Iterations")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def run_gd_baselines(iters=120):
    X, Y, _mse, _grad, _ = setup_benchmark_A()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)
    gd_steps_1, mag_1 = gradient_descent(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.08, iters=iters
    )
    gd_losses_1 = [mse(s) for s in gd_steps_1]

    net, grad_fn, _, _, _ = setup_benchmark_B()
    gd_steps_2, mag_2 = gradient_descent(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.06, iters=iters
    )
    gd_losses_2 = [net(s) for s in gd_steps_2]

    rosenbrock, grad_fn, _ = setup_benchmark_C()
    gd_steps_3, mag_3 = gradient_descent(
        grad_fn=grad_fn, x0=[-1.25, 0.5], alpha=0.001, iters=iters
    )
    gd_losses_3 = [rosenbrock(s) for s in gd_steps_3]

    return (
        [gd_losses_1, gd_losses_2, gd_losses_3],
        gd_steps_2,
        gd_steps_3,
        [mag_1, mag_2, mag_3],
    )


def question_1_I():
    gd_losses, gd_steps_2, gd_steps_3, _ = run_gd_baselines()

    X, Y, _mse, _grad, _ = setup_benchmark_A()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)
    steps_1, alphas_1 = polyak(
        loss_fn=mse, grad_fn=grad_fn, x0=[0, 0], eps=1e-4, f_star=0, iters=120
    )
    losses_1 = [mse(s) for s in steps_1]

    net, grad_fn, _, _, _ = setup_benchmark_B()
    steps_2, alphas_2 = polyak(
        loss_fn=net, grad_fn=grad_fn, x0=[0, 0], eps=1e-4, f_star=0, iters=120
    )
    losses_2 = [net(s) for s in steps_2]

    rosenbrock, grad_fn, _ = setup_benchmark_C()
    steps_3, alphas_3 = polyak(
        loss_fn=rosenbrock,
        grad_fn=grad_fn,
        x0=[-1.25, 0.5],
        eps=1e-3,
        f_star=0,
        iters=120,
    )
    losses_3 = [rosenbrock(s) for s in steps_3]

    plot_loss_subplots(
        [losses_1, losses_2, losses_3],
        "Polyak",
        "./final/images/question_1_I_losses.png",
        gd_losses_per_bench=gd_losses,
    )
    plot_contour_subplots(
        steps_2,
        steps_3,
        "Polyak",
        "./final/images/question_1_I_contours.png",
        gd_steps_b2=gd_steps_2,
        gd_steps_b3=gd_steps_3,
    )
    plot_scalar_alpha_subplots(
        [alphas_1, alphas_2, alphas_3],
        "Polyak",
        "./final/images/question_1_I_alphas.png",
    )


def question_1_II():
    gd_losses, gd_steps_2, gd_steps_3, _ = run_gd_baselines()

    X, Y, _mse, _grad, _ = setup_benchmark_A()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)
    steps_1, alphas_1 = adagrad(grad_fn=grad_fn, x0=[0, 0], alpha0=1.8, iters=120)
    losses_1 = [mse(s) for s in steps_1]

    net, grad_fn, _, _, _ = setup_benchmark_B()
    steps_2, alphas_2 = adagrad(grad_fn=grad_fn, x0=[0, 0], alpha0=1.2, iters=120)
    losses_2 = [net(s) for s in steps_2]

    rosenbrock, grad_fn, _ = setup_benchmark_C()
    steps_3, alphas_3 = adagrad(
        grad_fn=grad_fn, x0=[-1.25, 0.5], alpha0=0.45, iters=120
    )
    losses_3 = [rosenbrock(s) for s in steps_3]

    plot_loss_subplots(
        [losses_1, losses_2, losses_3],
        "Adagrad",
        "./final/images/question_1_II_losses.png",
        gd_losses_per_bench=gd_losses,
    )
    plot_contour_subplots(
        steps_2,
        steps_3,
        "Adagrad",
        "./final/images/question_1_II_contours.png",
        gd_steps_b2=gd_steps_2,
        gd_steps_b3=gd_steps_3,
    )
    plot_vector_alpha_subplots(
        [alphas_1, alphas_2, alphas_3],
        "Adagrad",
        "./final/images/question_1_II_alphas.png",
    )


def question_1_III():
    gd_losses, gd_steps_2, gd_steps_3, _ = run_gd_baselines()

    X, Y, _mse, _grad, _ = setup_benchmark_A()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)
    steps_1, alphas_1 = rms_prop(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.22, beta=0.9, iters=120
    )
    losses_1 = [mse(s) for s in steps_1]

    net, grad_fn, _, _, _ = setup_benchmark_B()
    steps_2, alphas_2 = rms_prop(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.14, beta=0.9, iters=120
    )
    losses_2 = [net(s) for s in steps_2]

    rosenbrock, grad_fn, _ = setup_benchmark_C()
    steps_3, alphas_3 = rms_prop(
        grad_fn=grad_fn,
        x0=[-1.25, 0.5],
        alpha=0.0035,
        beta=0.9,
        iters=120,
    )
    losses_3 = [rosenbrock(s) for s in steps_3]

    plot_loss_subplots(
        [losses_1, losses_2, losses_3],
        "RMSProp",
        "./final/images/question_1_III_losses.png",
        gd_losses_per_bench=gd_losses,
    )
    plot_contour_subplots(
        steps_2,
        steps_3,
        "RMSProp",
        "./final/images/question_1_III_contours.png",
        gd_steps_b2=gd_steps_2,
        gd_steps_b3=gd_steps_3,
    )
    plot_vector_alpha_subplots(
        [alphas_1, alphas_2, alphas_3],
        "RMSProp",
        "./final/images/question_1_III_alphas.png",
    )


def question_1_IV():
    gd_losses, gd_steps_2, gd_steps_3, _ = run_gd_baselines()

    X, Y, _mse, _grad, _ = setup_benchmark_A()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)
    steps_1, _ = heavy_ball(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.045, beta=0.88, iters=120
    )
    losses_1 = [mse(s) for s in steps_1]

    net, grad_fn, _, _, _ = setup_benchmark_B()
    steps_2, _ = heavy_ball(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.035, beta=0.90, iters=120
    )
    losses_2 = [net(s) for s in steps_2]

    rosenbrock, grad_fn, _ = setup_benchmark_C()
    steps_3, _ = heavy_ball(
        grad_fn=grad_fn,
        x0=[-1.25, 0.5],
        alpha=0.0008,
        beta=0.86,
        iters=120,
    )
    losses_3 = [rosenbrock(s) for s in steps_3]

    plot_loss_subplots(
        [losses_1, losses_2, losses_3],
        "Heavy Ball",
        "./final/images/question_1_IV_losses.png",
        gd_losses_per_bench=gd_losses,
    )
    plot_contour_subplots(
        steps_2,
        steps_3,
        "Heavy Ball",
        "./final/images/question_1_IV_contours.png",
        gd_steps_b2=gd_steps_2,
        gd_steps_b3=gd_steps_3,
    )


def question_2_I():
    gd_losses, gd_steps_2, gd_steps_3, _ = run_gd_baselines(iters=150)

    X, Y, _mse, _grad, _ = setup_benchmark_A()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)
    steps_1 = nesterov_momentum(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.08, beta_max=0.9, iters=150
    )
    losses_1 = [mse(s) for s in steps_1]

    net, grad_fn, _, _, _ = setup_benchmark_B()
    steps_2 = nesterov_momentum(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.035, beta_max=0.92, iters=150
    )
    losses_2 = [net(s) for s in steps_2]

    rosenbrock, grad_fn, _ = setup_benchmark_C()
    steps_3 = nesterov_momentum(
        grad_fn=grad_fn,
        x0=[-1.25, 0.5],
        alpha=0.0007,
        beta_max=0.9,
        iters=150,
    )
    losses_3 = [rosenbrock(s) for s in steps_3]

    plot_loss_subplots(
        [losses_1, losses_2, losses_3],
        "Nesterov",
        "./final/images/question_2_I_losses.png",
        gd_losses_per_bench=gd_losses,
    )
    plot_contour_subplots(
        steps_2,
        steps_3,
        "Nesterov",
        "./final/images/question_2_I_contours.png",
        gd_steps_b2=gd_steps_2,
        gd_steps_b3=gd_steps_3,
    )


def question_2_II():
    gd_losses, gd_steps_2, gd_steps_3, _ = run_gd_baselines(iters=150)

    X, Y, _mse, _grad, _ = setup_benchmark_A()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)
    steps_1 = adam(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.12, beta_1=0.82, beta_2=0.999, iters=150
    )
    losses_1 = [mse(s) for s in steps_1]

    net, grad_fn, _, _, _ = setup_benchmark_B()
    steps_2 = adam(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.08, beta_1=0.82, beta_2=0.999, iters=150
    )
    losses_2 = [net(s) for s in steps_2]

    rosenbrock, grad_fn, _ = setup_benchmark_C()
    steps_3 = adam(
        grad_fn=grad_fn,
        x0=[-1.25, 0.5],
        alpha=0.006,
        beta_1=0.8,
        beta_2=0.999,
        iters=150,
    )
    losses_3 = [rosenbrock(s) for s in steps_3]

    plot_loss_subplots(
        [losses_1, losses_2, losses_3],
        "Adam",
        "./final/images/question_2_II_losses.png",
        gd_losses_per_bench=gd_losses,
    )
    plot_contour_subplots(
        steps_2,
        steps_3,
        "Adam",
        "./final/images/question_2_II_contours.png",
        gd_steps_b2=gd_steps_2,
        gd_steps_b3=gd_steps_3,
    )


def question_2_III():
    X, y, mse, grad_fn, _ = setup_benchmark_A()

    _, losses_batch_5 = stochastic_gradient_descent(
        loss_fn=mse,
        grad_fn=grad_fn,
        X=X,
        y=y,
        x0=[0, 0],
        alpha=0.06,
        max_epochs=50,
        batch_size=5,
    )
    _, losses_batch_40 = stochastic_gradient_descent(
        loss_fn=mse,
        grad_fn=grad_fn,
        X=X,
        y=y,
        x0=[0, 0],
        alpha=0.06,
        max_epochs=50,
        batch_size=40,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("SGD Batch Size 5")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("f(x) (MSE)")
    ax1.plot(range(len(losses_batch_5)), losses_batch_5, color="blue")
    ax1.set_yscale("log")
    ax1.grid(True)

    ax2.set_title("SGD Batch Size 40")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("f(x) (MSE)")
    ax2.plot(range(len(losses_batch_40)), losses_batch_40, color="red")
    ax2.set_yscale("log")
    ax2.grid(True)

    fig.suptitle("Linear Regression f(x) vs Iterations (SGD)")
    plt.tight_layout()
    plt.savefig("./final/images/question_2_III_losses.png")
    plt.show()


def question_2_IV():
    X, y, mse, grad_fn, _ = setup_benchmark_A(noise_scaler=6)

    _, losses_batch_5 = stochastic_gradient_descent(
        loss_fn=mse,
        grad_fn=grad_fn,
        X=X,
        y=y,
        x0=[0, 0],
        alpha=0.06,
        max_epochs=50,
        batch_size=5,
    )
    _, losses_batch_40 = stochastic_gradient_descent(
        loss_fn=mse,
        grad_fn=grad_fn,
        X=X,
        y=y,
        x0=[0, 0],
        alpha=0.06,
        max_epochs=50,
        batch_size=40,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("SGD Batch Size 5")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("f(x) (MSE)")
    ax1.plot(range(len(losses_batch_5)), losses_batch_5, color="blue")
    ax1.set_yscale("log")
    ax1.grid(True)

    ax2.set_title("SGD Batch Size 40")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("f(x) (MSE)")
    ax2.plot(range(len(losses_batch_40)), losses_batch_40, color="red")
    ax2.set_yscale("log")
    ax2.grid(True)

    fig.suptitle("Linear Regression f(x) vs Iterations (SGD) Noise Scaler 6")
    plt.tight_layout()
    plt.savefig("./final/images/question_2_IV_losses.png")
    plt.show()


def question_3_I():
    g = lambda x: x**4
    dgdx = lambda x: 4 * x**3
    d2gdx = lambda x: 12 * x**2
    x0 = 0.25
    first_order_approx = lambda x: g(x0) + dgdx(x0) * (x - x0)
    second_order_approx = lambda x: (
        g(x0) + dgdx(x0) * (x - x0) + 0.5 * d2gdx(0.25) * (x - x0) ** 2
    )

    space = np.linspace(-1, 1, 400)

    _, ax = plt.subplots()
    ax.set_title("First- and Second-Order Approximations of $g(x)=x^4$ at $x=0.25$")
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.plot(space, g(space), color="blue", label="$g(x)=x^4$")
    ax.plot(
        space,
        first_order_approx(space),
        color="purple",
        linestyle="--",
        label="First Order Approximation",
    )
    ax.plot(
        space,
        second_order_approx(space),
        color="red",
        linestyle="--",
        label="Second Order Approximation",
    )
    ax.scatter([x0], [g(x0)], color="red", zorder=5, label="x0")
    ax.legend()
    ax.grid(True)
    plt.savefig("./final/images/question_3_I_approximations.png")
    plt.show()


def question_3_II():
    gd_losses, gd_steps_2, gd_steps_3, magnitudes = run_gd_baselines(iters=80)

    X, Y, mse, grad_fn, hessian_fn = setup_benchmark_A()
    steps_1, magnitudes_mse = newton_update(
        grad_fn=lambda theta: grad_fn(X, theta, Y),
        x0=[0, 0],
        hessian_fn=hessian_fn,
        iters=20,
        alpha=1.0,
    )
    losses_1 = [mse(X, step, Y) for step in steps_1]

    net, grad_fn, hessian_fn, _, _ = setup_benchmark_B()
    steps_2, magnitudes_net = newton_update(
        grad_fn=grad_fn,
        x0=[0, 0],
        hessian_fn=hessian_fn,
        iters=20,
        alpha=0.85,
    )
    losses_2 = [net(s) for s in steps_2]

    rosenbrock, grad_fn, hessian_fn = setup_benchmark_C()
    steps_3, magnitudes_rosenbrock = newton_update(
        grad_fn=grad_fn,
        x0=[-1.25, 0.5],
        hessian_fn=hessian_fn,
        iters=20,
        alpha=0.22,
    )
    losses_3 = [rosenbrock(s) for s in steps_3]

    plot_loss_subplots(
        [losses_1, losses_2, losses_3],
        "Newton Update",
        "./final/images/question_3_II_losses.png",
        gd_losses_per_bench=gd_losses,
    )
    plot_contour_subplots(
        steps_2,
        steps_3,
        "Newton Update",
        "./final/images/question_3_II_contours.png",
        gd_steps_b2=gd_steps_2,
        gd_steps_b3=gd_steps_3,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    newton_mags = [magnitudes_mse, magnitudes_net, magnitudes_rosenbrock]
    for ax, gd_mag, new_mag, label in zip(
        axes, magnitudes, newton_mags, BENCHMARK_LABELS
    ):
        ax.set_title(label)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Step Magnitude")
        ax.plot(range(len(gd_mag)), gd_mag, color="green", label="GD")
        ax.plot(range(len(new_mag)), new_mag, color="blue", label="Newton")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)
    fig.suptitle("Newton Update vs GD: Step Magnitude vs Iterations")
    plt.tight_layout()
    plt.savefig("./final/images/question_3_II_magnitudes.png")
    plt.show()


def question_4a_I():
    gd_losses, gd_steps_2, _, _ = run_gd_baselines(iters=120)

    net, _, _, finite_diff, _ = setup_benchmark_B()
    steps_delta_good, _ = gradient_descent(
        grad_fn=lambda x: finite_diff(x, delta=0.05),
        x0=[0, 0],
        iters=120,
        alpha=0.08,
    )
    losses_delta_good = [net(s) for s in steps_delta_good]
    steps_delta_poor, _ = gradient_descent(
        grad_fn=lambda x: finite_diff(x, delta=0.8),
        x0=[0, 0],
        iters=120,
        alpha=0.08,
    )
    losses_delta_poor = [net(s) for s in steps_delta_poor]

    _, ax = plt.subplots()
    ax.set_title("Toy Neural Net f(x) vs Iterations (Finite Difference)")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("f(x) Neural Net")
    ax.plot(range(len(gd_losses[1])), gd_losses[1], color="green", label="GD")
    ax.plot(
        range(len(losses_delta_good)),
        losses_delta_good,
        color="blue",
        label="Finite Diff δ=0.05",
    )
    ax.plot(
        range(len(losses_delta_poor)),
        losses_delta_poor,
        color="orange",
        label="Finite Diff δ=0.8",
    )
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    plt.savefig("./final/images/question_4a_I_losses.png")
    plt.show()

    gd_traj = np.array([np.array(s).flatten() for s in gd_steps_2])
    traj_good = np.array([np.array(s).flatten() for s in steps_delta_good])
    traj_poor = np.array([np.array(s).flatten() for s in steps_delta_poor])
    all_traj = np.vstack([gd_traj, traj_good, traj_poor])
    x1_min = np.min(all_traj[:, 0]) * 1.05
    x1_max = np.max(all_traj[:, 0]) * 1.05
    x2_min = np.min(all_traj[:, 1]) * 1.05
    x2_max = np.max(all_traj[:, 1]) * 1.05
    space_x1 = np.linspace(min(x1_min, -2), max(x1_max, 4), 200)
    space_x2 = np.linspace(min(x2_min, -1), max(x2_max, 5), 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = (X1 - 1) ** 2 + 5 * (X2 - 2) ** 2 + np.sin(X1)
    _, ax = plt.subplots()
    ax.set_title("Toy Neural Net: Finite Difference Contour Trajectories")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.contour(X1, X2, Z, levels=30)
    ax.plot(
        gd_traj[:, 0],
        gd_traj[:, 1],
        color="green",
        marker="o",
        ms=4,
        label=f"GD ({gd_traj[-1, 0]:.2f}, {gd_traj[-1, 1]:.2f})",
    )
    ax.plot(
        traj_good[:, 0],
        traj_good[:, 1],
        color="blue",
        marker="o",
        ms=4,
        label=f"Finite Diff δ=0.05 ({traj_good[-1, 0]:.2f}, {traj_good[-1, 1]:.2f})",
    )
    ax.plot(
        traj_poor[:, 0],
        traj_poor[:, 1],
        color="orange",
        marker="o",
        ms=4,
        label=f"Finite Diff δ=0.8 ({traj_poor[-1, 0]:.2f}, {traj_poor[-1, 1]:.2f})",
    )
    ax.legend()
    ax.grid(True)
    plt.savefig("./final/images/question_4a_I_contours.png")
    plt.show()


def question_4a_II():
    gd_losses, gd_steps_2, _, _ = run_gd_baselines(iters=220)

    net, _, _, _, nesterov_random_dir = setup_benchmark_B()
    steps_delta_good, _ = gradient_descent(
        grad_fn=lambda x: nesterov_random_dir(x, delta=0.05),
        x0=[0, 0],
        iters=220,
        alpha=0.025,
    )
    losses_delta_good = [net(s) for s in steps_delta_good]

    _, ax = plt.subplots()
    ax.set_title("Toy Neural Net f(x) vs Iterations (Nesterov Random Direction)")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("f(x) Neural Net")
    ax.set_yscale("log")
    ax.plot(range(len(gd_losses[1])), gd_losses[1], color="green", label="GD")
    ax.plot(
        range(len(losses_delta_good)),
        losses_delta_good,
        color="blue",
        label="Nesterov δ=0.05",
    )
    ax.legend()
    ax.grid(True)
    plt.savefig("./final/images/question_4a_II_losses.png")
    plt.show()

    gd_traj = np.array([np.array(s).flatten() for s in gd_steps_2])
    traj_good = np.array([np.array(s).flatten() for s in steps_delta_good])
    all_traj = np.vstack([gd_traj, traj_good])
    x1_min = np.min(all_traj[:, 0]) * 1.05
    x1_max = np.max(all_traj[:, 0]) * 1.05
    x2_min = np.min(all_traj[:, 1]) * 1.05
    x2_max = np.max(all_traj[:, 1]) * 1.05
    space_x1 = np.linspace(min(x1_min, -2), max(x1_max, 4), 200)
    space_x2 = np.linspace(min(x2_min, -1), max(x2_max, 5), 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = (X1 - 1) ** 2 + 5 * (X2 - 2) ** 2 + np.sin(X1)

    _, ax = plt.subplots()
    ax.set_title("Toy Neural Net: Nesterov Random Direction Contour Trajectories")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.contour(X1, X2, Z, levels=30)
    ax.plot(
        gd_traj[:, 0],
        gd_traj[:, 1],
        color="green",
        marker="o",
        ms=4,
        label=f"GD ({gd_traj[-1, 0]:.2f}, {gd_traj[-1, 1]:.2f})",
    )
    ax.plot(
        traj_good[:, 0],
        traj_good[:, 1],
        color="blue",
        marker="o",
        ms=4,
        label=f"Nesterov δ=0.05 ({traj_good[-1, 0]:.2f}, {traj_good[-1, 1]:.2f})",
    )
    ax.legend()
    ax.grid(True)
    plt.savefig("./final/images/question_4a_II_contours.png")
    plt.show()


def question_4b_I():
    gd_losses, _, gd_steps_3, _ = run_gd_baselines(iters=160)

    rosenbrock, _, _ = setup_benchmark_C()
    steps = nelder_mead(
        fn=rosenbrock,
        x0=[-1.25, 0.5],
        simplex_step=0.35,
        iters=160,
    )
    losses_nm = [rosenbrock(s) for s in steps]
    _, ax = plt.subplots()
    ax.set_title("Rosenbrock f(x) vs Iterations (Nelder-Mead)")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("f(x) Rosenbrock")
    ax.plot(range(len(gd_losses[2])), gd_losses[2], color="green", label="GD")
    ax.plot(range(len(losses_nm)), losses_nm, color="blue", label="Nelder-Mead")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    plt.savefig("./final/images/question_4b_I_losses.png")
    plt.show()

    traj_nm = np.array([np.array(s).flatten() for s in steps])
    gd_traj = np.array([np.array(s).flatten() for s in gd_steps_3])
    all_traj = np.vstack([gd_traj, traj_nm])
    x1_min = np.min(all_traj[:, 0]) * 1.05
    x1_max = np.max(all_traj[:, 0]) * 1.05
    x2_min = np.min(all_traj[:, 1]) * 1.05
    x2_max = np.max(all_traj[:, 1]) * 1.05
    space_x1 = np.linspace(min(x1_min, -1.5), max(x1_max, 2.0), 200)
    space_x2 = np.linspace(min(x2_min, -0.5), max(x2_max, 2.5), 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = (1 - X1) ** 2 + 100 * (X2 - X1**2) ** 2
    _, ax = plt.subplots()
    ax.set_title("Rosenbrock: Nelder-Mead Contour Trajectory")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.contour(X1, X2, Z, levels=30)
    ax.plot(
        gd_traj[:, 0],
        gd_traj[:, 1],
        color="green",
        marker="o",
        ms=4,
        label=f"GD ({gd_traj[-1, 0]:.2f}, {gd_traj[-1, 1]:.2f})",
    )
    ax.plot(
        traj_nm[:, 0],
        traj_nm[:, 1],
        color="blue",
        marker="o",
        ms=4,
        label=f"Nelder-Mead ({traj_nm[-1, 0]:.2f}, {traj_nm[-1, 1]:.2f})",
    )
    ax.scatter(1, 1, c="black", zorder=5, label="Global min (1,1)")
    ax.legend()
    ax.grid(True)
    plt.savefig("./final/images/question_4b_I_contours.png")
    plt.show()


def question_4b_II():
    rosenbrock, _, _ = setup_benchmark_C()
    steps, fx_samples = grid_search(loss_fn=rosenbrock)

    fx_values = np.array([fx for fx, _ in fx_samples])
    best_idx = np.argmin(fx_values)
    best_coord = steps[best_idx]
    running_min = np.minimum.accumulate(fx_values)

    x1_coords = np.array([s[0] for s in steps])
    x2_coords = np.array([s[1] for s in steps])
    x1_min, x1_max = np.min(x1_coords) * 1.05, np.max(x1_coords) * 1.05
    x2_min, x2_max = np.min(x2_coords) * 1.05, np.max(x2_coords) * 1.05
    space_x1 = np.linspace(min(x1_min, -1.5), max(x1_max, 2.0), 200)
    space_x2 = np.linspace(min(x2_min, -0.5), max(x2_max, 2.5), 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = (1 - X1) ** 2 + 100 * (X2 - X1**2) ** 2

    _, ax = plt.subplots()
    ax.set_title("Rosenbrock: Grid Search Sampling Pattern")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.contour(X1, X2, Z, levels=30)

    ax.scatter(
        x1_coords,
        x2_coords,
        c="blue",
        s=5,
        marker="o",
        alpha=0.5,
        label=f"Grid points ({len(steps)} samples)",
        zorder=5,
    )

    ax.scatter(
        best_coord[0],
        best_coord[1],
        c="red",
        s=80,
        marker="x",
        linewidths=2,
        zorder=6,
        label=f"Best ({best_coord[0]:.2f}, {best_coord[1]:.2f})",
    )

    ax.scatter(1, 1, c="black", zorder=7, label="Global min (1,1)")
    ax.legend()
    ax.grid(True)
    plt.savefig("./final/images/question_4b_II_contours.png")
    plt.show()

    _, ax = plt.subplots()
    ax.set_title("Rosenbrock f(x) vs Samples (Grid Search)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("f(x) Rosenbrock")
    ax.plot(range(len(running_min)), running_min, color="blue", label="Best so far")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    plt.savefig("./final/images/question_4b_II_best_so_far.png")
    plt.show()


def question_5_I():
    net, grad_fn, _, _, _ = setup_benchmark_B()

    net_penalised = lambda x, lam: net(x) + lam * max(0, -x[0, 0] + 0.5)

    def grad_fn_penalised(x, lam):
        grads = grad_fn(x)
        grad_x1, grad_x2 = grads[0, 0], grads[1, 0]
        x1 = x[0, 0]

        if x1 < 0.5:
            grad_x1 -= lam

        return np.array([[grad_x1], [grad_x2]])

    gd_steps_2, _ = gradient_descent(
        grad_fn=grad_fn, x0=[0.2, 4.0], alpha=0.07, iters=100
    )
    gd_losses_2 = [net(s) for s in gd_steps_2]

    steps_proj = projected_gradient_descent(
        grad_fn=grad_fn, alpha=0.08, x0=[0.2, 4.0], iters=100
    )
    proj_losses = [net(s) for s in steps_proj]

    steps_pen_low, _ = gradient_descent(
        grad_fn=lambda x: grad_fn_penalised(x, 0.15),
        x0=[0.2, 4.0],
        alpha=0.05,
        iters=100,
    )

    pen_losses_low = [net_penalised(s, 0.15) for s in steps_pen_low]

    steps_pen_mid, _ = gradient_descent(
        grad_fn=lambda x: grad_fn_penalised(x, 1.8),
        x0=[0.2, 4.0],
        alpha=0.05,
        iters=100,
    )
    pen_losses_mid = [net_penalised(s, 1.8) for s in steps_pen_mid]

    steps_pen_high, _ = gradient_descent(
        grad_fn=lambda x: grad_fn_penalised(x, 4.5),
        x0=[0.2, 4.0],
        alpha=0.03,
        iters=100,
    )
    pen_losses_high = [net_penalised(s, 4.5) for s in steps_pen_high]

    _, ax = plt.subplots()
    ax.set_title("Toy Neural Net f(x) vs Iterations (Projected GD)")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("f(x) Neural Net")
    ax.plot(range(len(gd_losses_2)), gd_losses_2, color="green", label="GD")
    ax.plot(range(len(proj_losses)), proj_losses, color="blue", label="Projected GD")
    ax.plot(range(len(pen_losses_low)), pen_losses_low, color="red", label="λ=0.15")
    ax.plot(range(len(pen_losses_mid)), pen_losses_mid, color="purple", label="λ=1.8")
    ax.plot(
        range(len(pen_losses_high)),
        pen_losses_high,
        color="orange",
        label="λ=4.5",
    )
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    plt.savefig("./final/images/question_5_I_losses.png")
    plt.show()

    gd_traj = np.array([np.array(s).flatten() for s in gd_steps_2])
    proj_traj = np.array([np.array(s).flatten() for s in steps_proj])
    pen_traj_low = np.array([np.array(s).flatten() for s in steps_pen_low])
    pen_traj_mid = np.array([np.array(s).flatten() for s in steps_pen_mid])
    pen_traj_high = np.array([np.array(s).flatten() for s in steps_pen_high])

    space_x1 = np.linspace(0.15, 0.7, 200)
    space_x2 = np.linspace(0, 5, 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = (X1 - 1) ** 2 + 5 * (X2 - 2) ** 2 + np.sin(X1)

    _, ax = plt.subplots()
    ax.set_title("Toy Neural Net: Contour andTrajectories")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.contour(X1, X2, Z, levels=20)
    rect = Rectangle(
        (0.5, 0),
        0.7 - 0.5,
        5,
        facecolor="blue",
        alpha=0.15,
        edgecolor="black",
        linestyle="--",
        linewidth=2,
        zorder=10,
        label="Feasible Region",
    )
    ax.add_patch(rect)
    ax.plot(
        gd_traj[:, 0],
        gd_traj[:, 1],
        color="green",
        marker="o",
        ms=4,
        label=f"GD ({gd_traj[-1, 0]:.2f}, {gd_traj[-1, 1]:.2f})",
    )
    ax.plot(
        proj_traj[:, 0],
        proj_traj[:, 1],
        color="blue",
        marker="o",
        ms=4,
        label=f"Penalised GD ({proj_traj[-1, 0]:.2f}, {proj_traj[-1, 1]:.2f})",
    )
    ax.plot(
        pen_traj_low[:, 0],
        pen_traj_low[:, 1],
        color="red",
        marker="o",
        ms=4,
        label=f"Penalised GD ({pen_traj_low[-1, 0]:.2f}, {pen_traj_low[-1, 1]:.2f})",
    )
    ax.plot(
        pen_traj_mid[:, 0],
        pen_traj_mid[:, 1],
        color="purple",
        marker="o",
        ms=4,
        label=f"Penalised GD ({pen_traj_mid[-1, 0]:.2f}, {pen_traj_mid[-1, 1]:.2f})",
    )
    ax.plot(
        pen_traj_high[:, 0],
        pen_traj_high[:, 1],
        color="orange",
        marker="o",
        ms=4,
        label=f"Penalised GD ({pen_traj_high[-1, 0]:.2f}, {pen_traj_high[-1, 1]:.2f})",
    )
    ax.legend()
    ax.grid(True)
    plt.savefig("./final/images/question_5_I_contours.png")
    plt.show()

    x1s_low = [step[0, 0] for step in steps_pen_low]
    x1s_mid = [step[0, 0] for step in steps_pen_mid]
    x1s_high = [step[0, 0] for step in steps_pen_high]
    x1s_proj = [step[0, 0] for step in steps_proj]

    violations_low = [max(0, 0.5 - x1) for x1 in x1s_low]
    violations_mid = [max(0, 0.5 - x1) for x1 in x1s_mid]
    violations_high = [max(0, 0.5 - x1) for x1 in x1s_high]
    violations_proj = [max(0, 0.5 - x1) for x1 in x1s_proj]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Constraint Violation (x1 >= 0.5) vs Iterations")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Violation")
    ax1.plot(
        range(len(violations_proj)),
        violations_proj,
        color="purple",
        label="Projected GD",
    )
    ax1.plot(range(len(violations_low)), violations_low, color="blue", label="λ=0.15")
    ax1.plot(range(len(violations_mid)), violations_mid, color="orange", label="λ=1.8")
    ax1.plot(range(len(violations_high)), violations_high, color="red", label="λ=4.5")
    ax1.set_yscale("symlog")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Constraint Violation (x1 >= 0.5) — First 30 Iterations")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Violation")
    ax2.plot(range(30), violations_proj[:30], color="purple", label="Projected GD")
    ax2.plot(range(30), violations_low[:30], color="blue", label="λ=0.15")
    ax2.plot(range(30), violations_mid[:30], color="orange", label="λ=1.8")
    ax2.plot(range(30), violations_high[:30], color="red", label="λ=4.5")
    ax2.set_yscale("symlog")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle("Constraint Violation (x1 >= 0.5) vs Iterations")
    plt.tight_layout()
    plt.savefig("./final/images/question_5_II_violations.png")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists("./final/images"):
        os.makedirs("./final/images")

    # question_1_I()
    # question_1_II()
    # question_1_III()
    # question_1_IV()
    #
    # question_2_I()
    # question_2_II()
    # question_2_III()
    # question_2_IV()

    # question_3_I()
    # question_3_II()

    # question_4a_I()
    # question_4a_II()
    # question_4b_I()
    # question_4b_II()

    question_5_I()
