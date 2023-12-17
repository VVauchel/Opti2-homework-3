import numpy as np
import Fun_Jac_Hess_v2 as fun
import prova as pr
from sympy import symbols, lambdify, log, diff
import SPFM as SPFM


nine = 9
lambd = 5

A, B = SPFM.Read_Data(1*nine, 1*nine, 1*nine)

'''
# Number of features
n = len(A[0, :])
# Number of points in A
n_a = len(A)
# Number of points in B
n_b = len(B)

# Testing if Callables are correct...
# Define the variables of the problem
mu = symbols("mu")
h, c, p, s, t = pr.Define_symbols(n, n_a, n_b)

# Define the function to minimize at each step
F = p/mu + pr.Barrier(lambd, A, B)[0]
# Define it numerically
num_F = lambdify((mu, h, c, p, s, t), F, 'numpy')

# Define gradient of F
gradF = pr.Differenciate(F, n, n_a, n_b)
# Make the gradient a callable function
num_gradF = lambdify((mu, h, c, p, s, t), gradF, 'numpy')


# Define the Hessian of F
HessF = []
for i in range(len(gradF)):
    new_row = pr.Differenciate(gradF[i], n, n_a, n_b)
    HessF += [new_row]
# Make Hessian a callable function
num_HessF = lambdify((mu, h, c, p, s, t), HessF, 'numpy')

# Define feasible x0 and try to compute a damped newton step
#x0 = [0 for i in range(n)] + [0] + [5] + [2 for i in range(n_a)] + [2 for i in range(n_b)]
x0, mu_0, _ = load_x0()

# Extract variables
h = x0[:n]
c = x0[n]
p = x0[n + 1]
s = x0[n + 2:n + 2 + n_a]
t = x0[n + 2 + n_a:]


#print(num_HessF(1,h,c,p,s,t))
#print(np.array(num_HessF(1,h,c,p,s,t)))
H = np.array(num_HessF(1,h,c,p,s,t))
print(type(H))
print(np.array_equal(H, H.T))
print(np.linalg.eigvals(H))
'''

# Initial mu
mu = 1
#print(f'F = {num_F(mu, h, c, p, s, t)}')

# Initialize algorithm

#x0, mu_0, delta = SPFM.initialization(A, B, lambd=5)
#print(f'x0 = {x0[:10]}, delta = {delta}')

#SPFM.update_x0(A,B)
x = SPFM.short_path_method(A, B, lambd = 5, eps = 1e-3)

#x = SPFM.long_path_method(A, B, lambd = 5, eps = 1e-3)
