import os

import matplotlib.pyplot as plt
import numpy as np


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
    losses = [loss_fn(xi)]
    alphas = []

    for _ in range(iters):
        grads = grad_fn(xi)

        alpha = (loss_fn(xi) - f_star) / (np.sum(grads**2) + eps)
        xi = xi - alpha * grads

        steps.append(xi.copy())
        losses.append(loss_fn(xi))
        alphas.append(alpha)

    return steps, losses, alphas


def adagrad(loss_fn, grad_fn, x0, alpha0, iters):
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
    losses = [loss_fn(xi)]
    alphas = []
    for _ in range(iters):
        step, alpha_t = step_fn(xi)
        xi = xi - step

        steps.append(xi.copy())
        losses.append(loss_fn(xi))
        alphas.append(alpha_t)

    return steps, losses, alphas


def rms_prop(loss_fn, grad_fn, x0, alpha, beta, iters):
    a0 = alpha
    xi = np.array([x0]).reshape(-1, 1)
    steps = [xi]
    losses = [loss_fn(xi)]
    alphas = []

    accumulator = 0
    for _ in range(iters):
        grads = grad_fn(xi)
        accumulator = beta * accumulator + (1 - beta) * grads**2

        alpha = a0 / (np.sqrt(accumulator) + 10**-5)
        xi = xi - alpha * grads

        steps.append(xi.tolist())
        losses.append(loss_fn(xi))
        alphas.append(alpha)

    return steps, losses, alphas


def heavy_ball(loss_fn, grad_fn, alpha, beta, x0, iters):
    xi = np.array(x0).reshape(-1, 1)
    steps = [xi.copy()]
    losses = [loss_fn(xi)]
    alphas = []

    z = np.zeros_like(xi)
    for _ in range(iters):
        grads = grad_fn(xi)
        z = beta * z + alpha * grads
        xi = xi - z

        steps.append(xi.tolist())
        losses.append(loss_fn(xi))
        alphas.append(z)

    return steps, losses, alphas


def gradient_descent(loss_fn, grad_fn, alpha, x0, iters):
    xi = np.array(x0, dtype=np.dtypes.Float64DType).reshape(-1, 1)
    steps = [xi.copy()]
    losses = [loss_fn(xi)]

    for _ in range(iters):
        grads = grad_fn(xi)
        xi -= alpha * grads

        steps.append(xi.tolist())
        losses.append(loss_fn(xi))

    return steps, losses


def nesterov_momentum(loss_fn, grad_fn, x0, alpha, beta_max, iters):
    xi = np.array([x0]).reshape(-1, 1)

    z = 0

    def step_fn(x, k):
        nonlocal z
        beta = min((k - 1) / (k + 2), beta_max)

        z = beta * z - alpha * grad_fn(x + beta * z)
        return -z

    steps = [xi.copy()]
    losses = [loss_fn(xi)]

    for k in range(1, iters + 1):
        step = step_fn(xi, k)
        xi = xi - step

        steps.append(xi.copy())
        losses.append(loss_fn(xi))

    return steps, losses


def adam(loss_fn, grad_fn, x0, alpha, beta_1, beta_2, iters):
    xi = np.array([x0]).reshape(-1, 1)
    steps = [xi.copy()]
    losses = [loss_fn(xi)]

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
        losses.append(loss_fn(xi))

    return steps, losses


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


def plot_vector_alpha(alphas, title, savepath, param_names=("x1", "x2")):
    alpha_arr = np.array([a.flatten() for a in alphas])
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Alpha")
    ax.plot(
        range(len(alpha_arr)),
        alpha_arr[:, 0],
        marker="o",
        color="red",
        label=f"alpha {param_names[0]}",
    )
    ax.plot(
        range(len(alpha_arr)),
        alpha_arr[:, 1],
        marker="s",
        color="green",
        label=f"alpha {param_names[1]}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(savepath)
    plt.show()


def plot_contour_net(steps, title, savepath):
    space_x1 = np.linspace(-2, 4, 200)
    space_x2 = np.linspace(-1, 5, 200)
    X1, X2 = np.meshgrid(space_x1, space_x2)
    Z = (X1 - 1) ** 2 + 5 * (X2 - 2) ** 2 + np.sin(X1)

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
    ax.legend()
    plt.savefig(savepath)
    plt.show()


def plot_contour_rosenbrock(steps, title, savepath):
    space_x1 = np.linspace(-0.5, 2.0, 200)
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


def question_1_I():
    X, Y, _mse, _grad = setup_benchmark_1()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)

    _, losses, alphas = polyak(
        loss_fn=mse, grad_fn=grad_fn, x0=[0, 0], eps=10**-4, f_star=0, iters=120
    )
    plot_loss(
        losses,
        "Linear Regression f(x) vs Iterations (Polyak)",
        "f(x) (MSE)",
        "./final/images/question1_I_loss_mse.png",
    )
    plot_scalar_alpha(
        alphas,
        "Linear Regression: Alpha vs Iterations (Polyak)",
        "./final/images/question1_I_alpha_mse.png",
    )

    net, grad_fn = setup_benchmark_2()

    steps, losses, alphas = polyak(
        loss_fn=net, grad_fn=grad_fn, x0=[0, 0], eps=10**-4, f_star=0, iters=120
    )
    plot_loss(
        losses,
        "Toy Neural Net f(x) vs Iterations (Polyak)",
        "f(x) Neural Net",
        "./final/images/question1_I_loss_net.png",
    )
    plot_scalar_alpha(
        alphas,
        "Toy Neural Net: Alpha vs Iterations (Polyak)",
        "./final/images/question1_I_alpha_net.png",
    )
    plot_contour_net(
        steps,
        "Toy Neural Net: Contour and Polyak Trajectory",
        "./final/images/question1_I_contour_net.png",
    )

    rosenbrock, grad_fn = setup_benchmark_3()

    steps, losses, alphas = polyak(
        loss_fn=rosenbrock, grad_fn=grad_fn, x0=[0, 0], eps=10**-3, f_star=0, iters=120
    )
    plot_loss(
        losses,
        "Rosenbrock f(x) vs Iterations (Polyak)",
        "f(x) Rosenbrock",
        "./final/images/question1_I_loss_rosenbrock.png",
    )
    plot_scalar_alpha(
        alphas,
        "Rosenbrock: Alpha vs Iterations (Polyak)",
        "./final/images/question1_I_alpha_rosenbrock.png",
    )
    plot_contour_rosenbrock(
        steps,
        "Rosenbrock: Contour and Polyak Trajectory",
        "./final/images/question1_I_contour_rosenbrock.png",
    )


def question_1_II():
    X, Y, _mse, _grad = setup_benchmark_1()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)

    _, losses, alphas = adagrad(
        loss_fn=mse, grad_fn=grad_fn, x0=[0, 0], alpha0=1.8, iters=120
    )
    plot_loss(
        losses,
        "Linear Regression f(x) vs Iterations (Adagrad)",
        "f(x) (MSE)",
        "./final/images/question1_II_loss_mse.png",
    )
    plot_vector_alpha(
        alphas,
        "Linear Regression: Alpha vs Iterations (Adagrad)",
        "./final/images/question1_II_alpha_mse.png",
        param_names=("θ0", "θ1"),
    )

    net, grad_fn = setup_benchmark_2()

    steps, losses, alphas = adagrad(
        loss_fn=net, grad_fn=grad_fn, x0=[0, 0], alpha0=1.2, iters=120
    )
    plot_loss(
        losses,
        "Toy Neural Net f(x) vs Iterations (Adagrad)",
        "f(x) Neural Net",
        "./final/images/question1_II_loss_net.png",
    )
    plot_vector_alpha(
        alphas,
        "Toy Neural Net: Alpha vs Iterations (Adagrad)",
        "./final/images/question1_II_alpha_net.png",
    )
    plot_contour_net(
        steps,
        "Toy Neural Net: Contour and Adagrad Trajectory",
        "./final/images/question1_II_contour_net.png",
    )

    rosenbrock, grad_fn = setup_benchmark_3()

    steps, losses, alphas = adagrad(
        loss_fn=rosenbrock, grad_fn=grad_fn, x0=[0, 0], alpha0=0.45, iters=120
    )
    plot_loss(
        losses,
        "Rosenbrock f(x) vs Iterations (Adagrad)",
        "f(x) Rosenbrock",
        "./final/images/question1_II_loss_rosenbrock.png",
    )
    plot_vector_alpha(
        alphas,
        "Rosenbrock: Alpha vs Iterations (Adagrad)",
        "./final/images/question1_II_alpha_rosenbrock.png",
    )
    plot_contour_rosenbrock(
        steps,
        "Rosenbrock: Contour and Adagrad Trajectory",
        "./final/images/question1_II_contour_rosenbrock.png",
    )


def question_1_III():
    X, Y, _mse, _grad = setup_benchmark_1()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)

    _, losses, alphas = rms_prop(
        loss_fn=mse, grad_fn=grad_fn, x0=[0, 0], alpha=0.22, beta=0.9, iters=120
    )
    plot_loss(
        losses,
        "Linear Regression f(x) vs Iterations (RMSProp)",
        "f(x) (MSE)",
        "./final/images/question1_III_loss_mse.png",
    )
    plot_vector_alpha(
        alphas,
        "Linear Regression: Alpha vs Iterations (RMSProp)",
        "./final/images/question1_III_alpha_mse.png",
        param_names=("θ0", "θ1"),
    )

    net, grad_fn = setup_benchmark_2()

    steps, losses, alphas = rms_prop(
        loss_fn=net, grad_fn=grad_fn, x0=[0, 0], alpha=0.14, beta=0.9, iters=120
    )
    plot_loss(
        losses,
        "Toy Neural Net f(x) vs Iterations (RMSProp)",
        "f(x) Neural Net",
        "./final/images/question1_III_loss_net.png",
    )
    plot_vector_alpha(
        alphas,
        "Toy Neural Net: Alpha vs Iterations (RMSProp)",
        "./final/images/question1_III_alpha_net.png",
    )
    plot_contour_net(
        steps,
        "Toy Neural Net: Contour and RMSProp Trajectory",
        "./final/images/question1_III_contour_net.png",
    )

    rosenbrock, grad_fn = setup_benchmark_3()

    steps, losses, alphas = rms_prop(
        loss_fn=rosenbrock,
        grad_fn=grad_fn,
        x0=[0, 0],
        alpha=0.0035,
        beta=0.9,
        iters=120,
    )
    plot_loss(
        losses,
        "Rosenbrock f(x) vs Iterations (RMSProp)",
        "f(x) Rosenbrock",
        "./final/images/question1_III_loss_rosenbrock.png",
    )
    plot_vector_alpha(
        alphas,
        "Rosenbrock: Alpha vs Iterations (RMSProp)",
        "./final/images/question1_III_alpha_rosenbrock.png",
    )
    plot_contour_rosenbrock(
        steps,
        "Rosenbrock: Contour and RMSProp Trajectory",
        "./final/images/question1_III_contour_rosenbrock.png",
    )


def question_1_IV():
    X, Y, _mse, _grad = setup_benchmark_1()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)

    _, losses, alphas = heavy_ball(
        loss_fn=mse, grad_fn=grad_fn, x0=[0, 0], alpha=0.045, beta=0.88, iters=120
    )
    plot_loss(
        losses,
        "Linear Regression f(x) vs Iterations (Heavy Ball)",
        "f(x) (MSE)",
        "./final/images/question1_IV_loss_mse.png",
    )
    plot_vector_alpha(
        alphas,
        "Linear Regression: Alpha vs Iterations (Heavy Ball)",
        "./final/images/question1_IV_alpha_mse.png",
        param_names=("θ0", "θ1"),
    )

    net, grad_fn = setup_benchmark_2()

    steps, losses, alphas = rms_prop(
        loss_fn=net, grad_fn=grad_fn, x0=[0, 0], alpha=0.035, beta=0.90, iters=120
    )
    plot_loss(
        losses,
        "Toy Neural Net f(x) vs Iterations (Heavy Ball)",
        "f(x) Neural Net",
        "./final/images/question1_III_loss_net.png",
    )
    plot_vector_alpha(
        alphas,
        "Toy Neural Net: Alpha vs Iterations (Heavy Ball)",
        "./final/images/question1_III_alpha_net.png",
    )
    plot_contour_net(
        steps,
        "Toy Neural Net: Contour and Heavy Ball Trajectory",
        "./final/images/question1_IV_contour_net.png",
    )

    rosenbrock, grad_fn = setup_benchmark_3()

    steps, losses, alphas = rms_prop(
        loss_fn=rosenbrock,
        grad_fn=grad_fn,
        x0=[0, 0],
        alpha=0.0008,
        beta=0.86,
        iters=120,
    )
    plot_loss(
        losses,
        "Rosenbrock f(x) vs Iterations (Heavy Ball)",
        "f(x) Rosenbrock",
        "./final/images/question1_IV_loss_rosenbrock.png",
    )
    plot_vector_alpha(
        alphas,
        "Rosenbrock: Alpha vs Iterations (Heavy Ball)",
        "./final/images/question1_IV_alpha_rosenbrock.png",
    )
    plot_contour_rosenbrock(
        steps,
        "Rosenbrock: Contour and Heavy Ball Trajectory",
        "./final/images/question1_IV_contour_rosenbrock.png",
    )


def baseline():
    X, Y, _mse, _grad = setup_benchmark_1()
    mse = lambda theta: _mse(X, theta, Y)
    grad_fn = lambda theta: _grad(X, theta, Y)

    steps, losses = gradient_descent(
        loss_fn=mse, grad_fn=grad_fn, x0=[0, 0], alpha=0.08, iters=120
    )
    plot_loss(
        losses,
        "Linear Regression f(x) vs Iterations (GD)",
        "f(x) (MSE)",
        "./final/images/baseline_loss_mse.png",
    )

    net, grad_fn = setup_benchmark_2()

    steps, losses = gradient_descent(
        loss_fn=net, grad_fn=grad_fn, x0=[0, 0], alpha=0.06, iters=120
    )
    plot_loss(
        losses,
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

    steps, losses = gradient_descent(
        loss_fn=rosenbrock,
        grad_fn=grad_fn,
        x0=[0, 0],
        alpha=0.0012,
        iters=120,
    )
    plot_loss(
        losses,
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

    steps, losses = nesterov_momentum(
        loss_fn=mse, grad_fn=grad_fn, x0=[0, 0], alpha=0.08, beta_max=0.9, iters=120
    )

    plot_loss(
        losses,
        "Linear Regression f(x) vs Iterations (Nesterov Momentum)",
        "f(x) (MSE)",
        "./final/images/question_2_I_loss_mse.png",
    )

    net, grad_fn = setup_benchmark_2()

    steps, losses = nesterov_momentum(
        loss_fn=net, grad_fn=grad_fn, x0=[0, 0], alpha=0.035, beta_max=0.92, iters=120
    )

    plot_loss(
        losses,
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

    steps, losses = nesterov_momentum(
        loss_fn=rosenbrock,
        grad_fn=grad_fn,
        x0=[0, 0],
        alpha=0.0007,
        beta_max=0.9,
        iters=120,
    )
    plot_loss(
        losses,
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

    steps, losses = adam(
        loss_fn=mse,
        grad_fn=grad_fn,
        x0=[0, 0],
        alpha=0.12,
        beta_1=0.82,
        beta_2=0.999,
        iters=150,
    )
    plot_loss(
        losses,
        "Linear Regression f(x) vs Iterations (Adam)",
        "f(x) (MSE)",
        "./final/images/question_2_II_loss_mse.png",
    )

    net, grad_fn = setup_benchmark_2()

    steps, losses = adam(
        loss_fn=net,
        grad_fn=grad_fn,
        x0=[0, 0],
        alpha=0.12,
        beta_1=0.82,
        beta_2=0.999,
        iters=150,
    )
    plot_loss(
        losses,
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

    steps, losses = adam(
        loss_fn=rosenbrock,
        grad_fn=grad_fn,
        x0=[0, 0],
        alpha=0.12,
        beta_1=0.82,
        beta_2=0.999,
        iters=150,
    )
    plot_loss(
        losses,
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

    baseline()
    question_1_I()
    question_1_II()
    question_1_III()
    question_1_IV()

    question_2_I()
    question_2_II()
    question_2_III()
    question_2_IV()
