import numpy as np
import Fun_Jac_Hess as fun
import prova as pr
from sympy import symbols, lambdify, log, diff
import SPFM as SPFM

A,B = pr.Read_Data(6,15,15)

lambd=5

# Number of features
n = len(A[0, :])
# Number of points in A
n_a = len(A)
# Number of points in B
n_b = len(B)

# Define the variables of the problem
mu = symbols("mu")
h, c, p, s, t = pr.Define_symbols(n,n_a,n_b)

# Define the function to minimize at each step
F = p/mu + pr.Barrier(lambd,A,B)[0]
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
x0 = [0 for i in range(n)] + [0] + [5] + [2 for i in range(n_a)] + [2 for i in range(n_b)]

# Extract variables
h = x0[:n]
c = x0[n]
p = x0[n + 1]
s = x0[n + 2:n + 2 + n_a]
t = x0[n + 2 + n_a:]

'''
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

#This while is the implementation of the initialization described in Lec10
delta = 1
while delta >= 1:
    #x0, delta = pr.damped_N(gradF, HessF, x0, n, n_a, n_b, mu)
    x0, delta = SPFM.damped_N(x0,A,B,lambd,mu=1)
    # Extract variables

    print(f'F = {fun.New_Obj(x0,1,5,A,B)}')
    print(f'delta = {delta}')
# We should get delta decreasing, but it's only correct for small number of features and points
