from abc import abstractclassmethod
from sympy import symbols, init_printing, diff, lambdify
from matplotlib import pyplot as plt
import numpy as np
import random
import math

init_printing(pretty_print=True, use_latex=False)  

### Question I
### part a
x = symbols('x', real=True)
f = x ** 4
dfdx = diff(f, x)
print(dfdx)

### part b
f = lambdify(x, f)
dfdx = lambdify(x, dfdx)

x_values = np.arange(-2, 2.1, 0.1)
y_values = [dfdx(x) for x in x_values]

plt.figure(figsize=(8,5))  
plt.plot(x_values, y_values, 'k--', label='Exact Derivative', linewidth=2.5);

finite_diff_approx = lambda  f, x, delta: (f(x + delta) - f(x)) / delta

y_values_approx = [finite_diff_approx(f, xs, delta=0.01) for xs in x_values]

plt.plot(x_values, y_values_approx,  label=f'Finite difference (δ={0.01})');

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
plt.show();


### part d
x = symbols('x')
f = x ** 4

def gradient_descent(function, input, iterations=100, alpha=0.05, initial_value= 1, verbose=False):
    if(verbose):
        print(f'Finding optimum of function {function}')
        print(f'initial value: {initial_value}')

    f = lambdify(input, function)
    dfdx = lambdify(input, diff(function, input))
    x = initial_value
    x_values = [x]
    function_values = [f(x)]

    for iteration in range(1, iterations+1):
        step = alpha * dfdx(x)
        x -= step   
        x_values.append(x)
        function_values.append(f(x))

        if(verbose and iteration % 10 == 0):
            print(f'Iteration: {iteration}, value: {x}')

    plt.plot(range(iterations+1), function_values, label='f(x)')
    plt.plot(range(iterations+1), x_values, label='x')
    plt.title(f'f(x) and x vs. iterations, {alpha=}')
    plt.xlabel('iterations')
    plt.ylabel('f(x) / x')
    plt.legend(loc='best', framealpha=0.95)
    plt.savefig(f'./images/question1e_alpha={alpha}.png', dpi=300, bbox_inches='tight')
    plt.show();
    return x

optimum = gradient_descent(f, x, iterations=20, alpha=0.05, initial_value=1)
print(f'{optimum=}, {0.05=}')

optimum = gradient_descent(f, x, iterations=20, alpha=0.5, initial_value=1)
print(f'{optimum=}, {0.5=}')

optimum = gradient_descent(f, x, iterations=20, alpha=1.2, initial_value=1)
print(f'{optimum=}, {1.2=}')















