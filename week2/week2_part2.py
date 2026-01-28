from sympy import symbols, init_printing, diff, lambdify
import numpy as np
import random
from matplotlib import pyplot as plt

init_printing(use_unicode=True, use_latex=False)  

x = symbols('x')

f = x ** 4

def gradient_descent(function, input, iterations=100, learning_rate=0.03):
    print(f'Finding optimum of function {function}')
    derivative = lambdify(input, diff(function, input))
    x = random.uniform(0, 4)
    steps = []
    for iteration in range(iterations):
        step = learning_rate * derivative(x)
        x -= step   
        steps.append(x)

        if(iteration % 10 == 0):
            print(f'Iteration: {iteration}, value: {x}')

    plt.plot(range(iterations), steps)
    plt.yscale('log')
    plt.show();
    return x

optimum = gradient_descent(f, x, iterations=100, learning_rate=0.001)

print(optimum)
