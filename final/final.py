import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["savefig.dpi"] = 150


def setup_benchmark_1(noise_scaler=1):
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

    return X, Y, mse, grad_fn


def setup_benchmark_2():

    def net(x):
        x1, x2 = x1, x2 = x[0, 0], x[1, 0]
        return ((x1 - 1) ** 2) + (5 * (x2 - 2) ** 2) + np.sin(x1)

    def grad_fn(x):
        x1, x2 = x1, x2 = x[0, 0], x[1, 0]
        dfdx1 = 2 * (x1 - 1) + np.cos(x1)
        dfdx2 = 10 * (x2 - 2)
        return np.array([dfdx1, dfdx2]).reshape(-1, 1)

    return net, grad_fn


def setup_benchmark_3():
    def rosenbrock(x):
        x1, x2 = x1, x2 = x[0, 0], x[1, 0]
        return ((1 - x1) ** 2) + (100 * (x2 - x1**2) ** 2)

    def grad_fn(x):
        x1, x2 = x1, x2 = x[0, 0], x[1, 0]
        dfdx1 = (-2 * (1 - x1)) - 400 * x1 * (x2 - x1**2)
        dfdx2 = 200 * (x2 - x1**2)
        return np.array([dfdx1, dfdx2]).reshape(-1, 1)

    return rosenbrock, grad_fn


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

    for _ in range(iters):
        grads = grad_fn(xi)
        xi -= alpha * grads

        steps.append(xi.copy())

    return steps


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


def plot_loss(losses, title, ylabel, savepath):
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel(ylabel)
    ax.plot(range(len(losses)), losses, color="blue", label="Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(savepath)
    plt.show()


def plot_scalar_alpha(alphas, title, savepath):
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Alpha")
    ax.plot(range(len(alphas)), alphas, color="blue", marker="o", label="Alpha")
    ax.legend()
    ax.grid(True, alpha=0.3)
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
    ax.contour(X1, X2, Z, levels=30, cmap="plasma")
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
    ax.contour(X1, X2, Z, levels=30, cmap="plasma")
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


_BENCH_LABELS = ["Linear Regression", "Toy Neural Net", "Rosenbrock"]
_BENCH_YLABELS = ["f(x) (MSE)", "f(x) Neural Net", "f(x) Rosenbrock"]


def plot_loss_subplots(
    losses_per_bench, optimizer_name, savepath, gd_losses_per_bench=None
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, losses, label, ylabel in zip(
        axes, losses_per_bench, _BENCH_LABELS, _BENCH_YLABELS
    ):
        ax.set_title(label)
        ax.set_xlabel("Iterations")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    if gd_losses_per_bench is not None:
        for ax, gd_losses in zip(axes, gd_losses_per_bench):
            ax.plot(range(len(gd_losses)), gd_losses, color="green", label="GD")
    for ax, losses in zip(axes, losses_per_bench):
        ax.plot(range(len(losses)), losses, color="blue", label=optimizer_name)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(f"{optimizer_name}: Loss vs Iterations")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def plot_contour_subplots(
    steps_b2, steps_b3, optimizer_name, savepath, gd_steps_b2=None, gd_steps_b3=None
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
    ax1.contour(X1, X2, Z, levels=30, cmap="plasma")
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
    ax1.legend(fontsize=8)

    sx1 = np.linspace(-1.5, 2.0, 200)
    sx2 = np.linspace(-0.5, 2.5, 200)
    X1, X2 = np.meshgrid(sx1, sx2)
    Z = (1 - X1) ** 2 + 100 * (X2 - X1**2) ** 2
    ax2.contour(X1, X2, Z, levels=30, cmap="plasma")
    if gd_steps_b3 is not None:
        gd_traj = np.array([np.array(s).flatten() for s in gd_steps_b3])
        ax2.plot(
            gd_traj[:, 0],
            gd_traj[:, 1],
            color="green",
            marker="o",
            ms=4,
            label=f"GD ({gd_traj[-1, 0]:.2f}, {gd_traj[-1, 1]:.2f})",
        )
    traj = np.array([np.array(s).flatten() for s in steps_b3])
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
    ax2.legend(fontsize=8)

    fig.suptitle(f"{optimizer_name}: Contour Trajectories")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def plot_scalar_alpha_subplots(alphas_per_bench, optimizer_name, savepath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, alphas, label in zip(axes, alphas_per_bench, _BENCH_LABELS):
        ax.set_title(label)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Alpha")
        ax.plot(range(len(alphas)), alphas, color="blue")
        ax.grid(True, alpha=0.3)
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
        axes, alphas_per_bench, _BENCH_LABELS, param_names_per_bench
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
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{optimizer_name}: Alpha vs Iterations")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


def _run_gd_baselines():
    X, Y, _mse, _grad = setup_benchmark_1()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)
    gd_steps_1 = gradient_descent(grad_fn=grad_fn, x0=[0, 0], alpha=0.08, iters=120)
    gd_losses_1 = [mse(s) for s in gd_steps_1]

    net, grad_fn = setup_benchmark_2()
    gd_steps_2 = gradient_descent(grad_fn=grad_fn, x0=[0, 0], alpha=0.06, iters=120)
    gd_losses_2 = [net(s) for s in gd_steps_2]

    rosenbrock, grad_fn = setup_benchmark_3()
    gd_steps_3 = gradient_descent(
        grad_fn=grad_fn, x0=[-1.25, 0.5], alpha=0.0012, iters=120
    )
    gd_losses_3 = [rosenbrock(s) for s in gd_steps_3]

    return [gd_losses_1, gd_losses_2, gd_losses_3], gd_steps_2, gd_steps_3


def question_1_I():
    gd_losses, gd_steps_2, gd_steps_3 = _run_gd_baselines()

    X, Y, _mse, _grad = setup_benchmark_1()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)
    steps_1, alphas_1 = polyak(
        loss_fn=mse, grad_fn=grad_fn, x0=[0, 0], eps=1e-4, f_star=0, iters=120
    )
    losses_1 = [mse(s) for s in steps_1]

    net, grad_fn = setup_benchmark_2()
    steps_2, alphas_2 = polyak(
        loss_fn=net, grad_fn=grad_fn, x0=[0, 0], eps=1e-4, f_star=0, iters=120
    )
    losses_2 = [net(s) for s in steps_2]

    rosenbrock, grad_fn = setup_benchmark_3()
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
    gd_losses, gd_steps_2, gd_steps_3 = _run_gd_baselines()

    X, Y, _mse, _grad = setup_benchmark_1()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)
    steps_1, alphas_1 = adagrad(grad_fn=grad_fn, x0=[0, 0], alpha0=1.8, iters=120)
    losses_1 = [mse(s) for s in steps_1]

    net, grad_fn = setup_benchmark_2()
    steps_2, alphas_2 = adagrad(grad_fn=grad_fn, x0=[0, 0], alpha0=1.2, iters=120)
    losses_2 = [net(s) for s in steps_2]

    rosenbrock, grad_fn = setup_benchmark_3()
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
    gd_losses, gd_steps_2, gd_steps_3 = _run_gd_baselines()

    X, Y, _mse, _grad = setup_benchmark_1()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)
    steps_1, alphas_1 = rms_prop(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.22, beta=0.9, iters=120
    )
    losses_1 = [mse(s) for s in steps_1]

    net, grad_fn = setup_benchmark_2()
    steps_2, alphas_2 = rms_prop(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.14, beta=0.9, iters=120
    )
    losses_2 = [net(s) for s in steps_2]

    rosenbrock, grad_fn = setup_benchmark_3()
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
    gd_losses, gd_steps_2, gd_steps_3 = _run_gd_baselines()

    X, Y, _mse, _grad = setup_benchmark_1()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)
    steps_1, _ = heavy_ball(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.045, beta=0.88, iters=120
    )
    losses_1 = [mse(s) for s in steps_1]

    net, grad_fn = setup_benchmark_2()
    steps_2, _ = heavy_ball(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.035, beta=0.90, iters=120
    )
    losses_2 = [net(s) for s in steps_2]

    rosenbrock, grad_fn = setup_benchmark_3()
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


def baseline():
    X, Y, _mse, _grad = setup_benchmark_1()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)

    steps = gradient_descent(grad_fn=grad_fn, x0=[0, 0], alpha=0.08, iters=120)
    plot_loss(
        [mse(s) for s in steps],
        "Linear Regression f(x) vs Iterations (GD)",
        "f(x) (MSE)",
        "./final/images/baseline_loss_mse.png",
    )

    net, grad_fn = setup_benchmark_2()

    steps = gradient_descent(grad_fn=grad_fn, x0=[0, 0], alpha=0.06, iters=120)
    plot_loss(
        [net(s) for s in steps],
        "Toy Neural Net f(x) vs Iterations (GD)",
        "f(x) Neural Net",
        "./final/images/basline_loss_net.png",
    )
    plot_contour_net(
        steps,
        "Toy Neural Net: Contour and Baseline GD Trajectory",
        "./final/images/baseline_contour_net.png",
    )

    rosenbrock, grad_fn = setup_benchmark_3()

    steps = gradient_descent(
        grad_fn=grad_fn,
        x0=[-1.25, 0.5],
        alpha=0.0012,
        iters=120,
    )
    plot_loss(
        [rosenbrock(s) for s in steps],
        "Rosenbrock f(x) vs Iterations (GD)",
        "f(x) Rosenbrock",
        "./final/images/baseline_loss_rosenbrock.png",
    )
    plot_contour_rosenbrock(
        steps,
        "Rosenbrock: Contour and Baseline GD Trajectory",
        "./final/images/baseline_contour_rosenbrock.png",
    )


def question_2_I():
    X, Y, _mse, _grad = setup_benchmark_1()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)

    steps = nesterov_momentum(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.08, beta_max=0.9, iters=120
    )
    plot_loss(
        [mse(s) for s in steps],
        "Linear Regression f(x) vs Iterations (Nesterov Momentum)",
        "f(x) (MSE)",
        "./final/images/question_2_I_loss_mse.png",
    )

    net, grad_fn = setup_benchmark_2()

    steps = nesterov_momentum(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.035, beta_max=0.92, iters=120
    )
    plot_loss(
        [net(s) for s in steps],
        "Toy Neural Net f(x) vs Iterations (Nesterov Momentum)",
        "f(x) Neural Net",
        "./final/images/question_2_I_loss_net.png",
    )
    plot_contour_net(
        steps,
        "Toy Neural Net: Contour and Nesterov Momentum Trajectory",
        "./final/images/question_2_I_contour_net.png",
    )

    rosenbrock, grad_fn = setup_benchmark_3()

    steps = nesterov_momentum(
        grad_fn=grad_fn,
        x0=[-1.25, 0.5],
        alpha=0.0007,
        beta_max=0.9,
        iters=120,
    )
    plot_loss(
        [rosenbrock(s) for s in steps],
        "Rosenbrock f(x) vs Iterations (Nesterov Momentum)",
        "f(x) Rosenbrock",
        "./final/images/question_2_I_loss_rosenbrock.png",
    )
    plot_contour_rosenbrock(
        steps,
        "Rosenbrock: Contour and Nesterov Momentum Trajectory",
        "./final/images/question_2_I_contour_rosenbrock.png",
    )


def question_2_II():
    X, Y, _mse, _grad = setup_benchmark_1()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)

    steps = adam(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.12, beta_1=0.82, beta_2=0.999, iters=150
    )
    plot_loss(
        [mse(s) for s in steps],
        "Linear Regression f(x) vs Iterations (Adam)",
        "f(x) (MSE)",
        "./final/images/question_2_II_loss_mse.png",
    )

    net, grad_fn = setup_benchmark_2()

    steps = adam(
        grad_fn=grad_fn, x0=[0, 0], alpha=0.12, beta_1=0.82, beta_2=0.999, iters=150
    )
    plot_loss(
        [net(s) for s in steps],
        "Toy Neural Net f(x) vs Iterations (Adam)",
        "f(x) Neural Net",
        "./final/images/question_2_II_loss_net.png",
    )
    plot_contour_net(
        steps,
        "Toy Neural Net: Contour and Adam Trajectory",
        "./final/images/question_2_II_contour_net.png",
    )

    rosenbrock, grad_fn = setup_benchmark_3()

    steps = adam(
        grad_fn=grad_fn,
        x0=[-1.25, 0.5],
        alpha=0.12,
        beta_1=0.82,
        beta_2=0.999,
        iters=150,
    )
    plot_loss(
        [rosenbrock(s) for s in steps],
        "Rosenbrock f(x) vs Iterations (Adam)",
        "f(x) Rosenbrock",
        "./final/images/question_2_II_loss_rosenbrock.png",
    )
    plot_contour_rosenbrock(
        steps,
        "Rosenbrock: Contour and Adam Trajectory",
        "./final/images/question_2_II_contour_rosenbrock.png",
    )


def question_2_III():
    X, y, mse, grad_fn = setup_benchmark_1()

    _, losses = stochastic_gradient_descent(
        loss_fn=mse,
        grad_fn=grad_fn,
        X=X,
        y=y,
        x0=[0, 0],
        alpha=0.06,
        max_epochs=50,
        batch_size=5,
    )
    plot_loss(
        losses,
        "Linear Regression f(x) vs Iterations (SGD) Batch Size 5",
        "f(x) (MSE)",
        "./final/images/question_2_III_loss_mse_batch_size_5.png",
    )

    _, losses = stochastic_gradient_descent(
        loss_fn=mse,
        grad_fn=grad_fn,
        X=X,
        y=y,
        x0=[0, 0],
        alpha=0.06,
        max_epochs=50,
        batch_size=40,
    )
    plot_loss(
        losses,
        "Linear Regression f(x) vs Iterations (SGD) Batch Size 40",
        "f(x) (MSE)",
        "./final/images/question_2_III_loss_mse_batch_size_40.png",
    )


def question_2_IV():
    X, y, mse, grad_fn = setup_benchmark_1(noise_scaler=6)

    _, losses = stochastic_gradient_descent(
        loss_fn=mse,
        grad_fn=grad_fn,
        X=X,
        y=y,
        x0=[0, 0],
        alpha=0.06,
        max_epochs=50,
        batch_size=5,
    )
    plot_loss(
        losses,
        "Linear Regression f(x) vs Iterations (SGD) Batch Size 5",
        "f(x) (MSE)",
        "./final/images/question_2_IV_loss_mse_batch_size_5.png",
    )

    _, losses = stochastic_gradient_descent(
        loss_fn=mse,
        grad_fn=grad_fn,
        X=X,
        y=y,
        x0=[0, 0],
        alpha=0.06,
        max_epochs=50,
        batch_size=40,
    )
    plot_loss(
        losses,
        "Linear Regression f(x) vs Iterations (SGD) Batch Size 40",
        "f(x) (MSE)",
        "./final/images/question_2_IV_loss_mse_batch_size_40.png",
    )


if __name__ == "__main__":
    if not os.path.exists("./final/images"):
        os.makedirs("./final/images")

    # baseline()
    # question_1_I()
    # question_1_II()
    # question_1_III()
    # question_1_IV()
    #
    question_2_I()
    # question_2_II()
    # question_2_III()
    # question_2_IV()
