from sympy import symbols, lambdify, log
import numpy as np

n = 10
n_a = 15
n_b = 20

lambd = 5

# Define the simbols:
h = symbols('h1:%d' % (n + 1))  # h as an n-dimensional vector
h = np.array(h)
print(h)
c, p = symbols('c p')  # c and p as scalars
s = symbols('s1:%d' % (n_a + 1))  # s as an n_a-dimensional vector
t = symbols('t1:%d' % (n_b + 1))  # t as an n_b-dimensional vector



