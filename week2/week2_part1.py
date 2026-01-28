from sympy import symbols, init_printing, diff, lambdify
from matplotlib import pyplot as plt
import numpy as np

init_printing(use_unicode=True, use_latex=False)  

x = symbols('x', real=True)
y = x ** 4
dx = diff(y, x)

f = lambdify(x, y)
df = lambdify(x, dx, 'numpy')

x_values = np.arange(-1, 1.1, 0.1)
y_values = [df(x) for x in x_values]

plt.figure(figsize=(8,5))  
plt.plot(x_values, y_values, 'k--', label='Exact Derivative', linewidth=3);

finite_diff_approx = lambda  f, x, delta: (f(x + delta) - f(x)) / delta

deltas = [1, 0.1, 0.01, 0.001]

for d in deltas:
    y_approx_values = [finite_diff_approx(f, x, d) for x in x_values]
    plt.plot(x_values, y_approx_values, label=f'Finite difference (δ={d})');


plt.title('Exact Derivative vs. Finite Differences', fontsize=16, pad=10)
plt.xlabel('x', fontsize=14)
plt.ylabel('Derivative', fontsize=14)
plt.grid(visible=True, alpha=0.4)
plt.tight_layout()
plt.legend(loc='best', framealpha=0.95)
plt.savefig('./images/delta_range_comparison.png', dpi=300, bbox_inches='tight')
plt.show();
