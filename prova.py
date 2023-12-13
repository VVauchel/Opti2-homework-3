import numpy as np
import dill
import matplotlib.pyplot as plt
import time
from sympy import symbols, lambdify, log, diff

image_size = 28  # width and length
no_of_different_labels = 10  #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "data/mnist/"
train_data = np.loadtxt(data_path + "mnist_train.csv",
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv",
                       delimiter=",")

# Extract 0s from the training set
A = train_data[train_data[:, 0] == 0.]

# Delete the first column
A = A[0:5, 1:]

# Only keep some columns such that first element is non zero (for start 10 of them)
non_zero_ind = np.nonzero(A[0, :])[0]
indices = np.random.choice(non_zero_ind, size=10, replace=False)
A = A[:, indices]

# Extract non-0s
B = train_data[train_data[:, 0] != 0.]
# Delete first column
B = B[0:5, 1:]
# Keep the same columns you kept in A
B = B[:, indices]


'''# Assuming you have loaded the MNIST data into A and B
# A contains zero digits, and B contains the other digits'''


# Set the regularization parameter lambda
lambd = float(5)

def short_path_method(A,B,lambd):
    '''This function implements the short path following method to optimize 
    the problem in Homework 3 of the course Optimization Models and methods II
    2023/2024
    Input:  lambd: regularization coefficient
            A: set 1, matrix with coordinates of points as rows 
            B: set 2, same as A
    '''

    # Number of features
    n = len(A[0, :])

    # Number of points in A
    n_a = len(A)

    # Number of points in B
    n_b = len(B)

    # Define the simbols:
    h = symbols('h1:%d' % (n + 1))  # h as an n-dimensional vector
    c, p = symbols('c p')  # c and p as scalars
    s = symbols('s1:%d' % (n_a + 1))  # s as an n_a-dimensional vector
    t = symbols('t1:%d' % (n_b + 1))  # t as an n_b-dimensional vector


    # Define the self concordant barrier with sympy
    F,v = Barrier(h, c, s, t, p)

    # Initialize with x_0 and mu_0
    x_0, mu_0 = initialization_sspf(F, n, n_a, n_b)

    # While cycle


    return 1

def Define_symbols(n, n_a, n_b):

    # Define the simbols:
    h = symbols('h1:%d' % (n + 1))  # h as an n-dimensional vector
    c, p = symbols('c p')  # c and p as scalars
    s = symbols('s1:%d' % (n_a + 1))  # s as an n_a-dimensional vector
    t = symbols('t1:%d' % (n_b + 1))  # t as an n_b-dimensional vector

    return h, c, p, s, t

def Barrier(lambd,n,n_a,n_b):
    '''Self concordant barrier'''

    # Define Symbols
    h, c, p, s, t = Define_symbols(n, n_a, n_b)


    # Define the self concordant barrier
    F = 0
    v = 0

    F = F - sum(log(s[i]) for i in range(n_a))
    v += n_a
    F = F - sum(log(t[i]) for i in range(n_b))
    v += n_b
    F = F - sum(log(-1 + s[i] - sum(h[j] * A[i, j] for j in range(n)) - c) for i in range(n_a))
    v += n_a
    F = F - sum(log(sum(h[j] * B[i, j] for j in range(n)) + c - 1 + t[i]) for i in range(n_b))
    v += n_b
    F = F - log(p - Objective(lambd, n, n_a, n_b))
    v += 1
    return F, v

def Objective(lambd,n,n_a,n_b):
    '''Defines the objective function of ex A3 as a sympy function'''

    # Define Symbols
    h, c, p, s, t = Define_symbols(n, n_a, n_b)

    O = lambd*sum((h[i]**2) for i in range(n))
    O = O + sum(s[i] for i in range(len(s)))/len(s)
    O = O + sum(t[i] for i in range(len(t)))/len(t)
    return G

def initialization_sspf(F, gradF, HessF, n, n_a, n_b):
    '''First draft: chose an interior point x_0 and find a mu_0 such that 
    delta_mu_0 < 1'''




    mu_0 = 1
    x_0 = 1


    return x_0, mu_0

    n = len(A[0, :])

    # Number of points in A
    n_a = len(A)

    # Number of points in B
    n_b = len(B)


#start_time = time.time()

    # Number of features
n = len(A[0, :])

    # Number of points in A
n_a = len(A)

    # Number of points in B
n_b = len(B)

# Define the variables of the problem
mu = symbols("mu")
h, c, p, s, t = Define_symbols(n, n_a, n_b)

# Define the function to minimize at each step
F = p/mu + Barrier(lambd, n, n_a, n_b)[0]

# Define it numerically
num_F = lambdify((mu, h, c, p, s, t), F, 'numpy')

# Save the function to a file
with open('my_function.pkl', 'wb') as file:
    dill.dump(num_F, file)

# Now, in a different script, you can load the function
with open('my_function.pkl', 'rb') as file:
    loaded_function = dill.load(file)

# Use the loaded function
print("Result using the loaded function:", loaded_function)

def Differenciate(func,n,n_a,n_b):
    '''Differenciate wrt (p, h, c, s, t)'''

    # Define symbols
    h, c, p, s, t = Define_symbols(n, n_a, n_b)

    # Define the gradient of func
    grad = [diff(func, p)]
    grad += [diff(func, h[i]) for i in range(n)]
    grad += [diff(func, c)]
    grad += [diff(func, s[i]) for i in range(n_a)]
    grad += [diff(func, t[i]) for i in range(n_b)]

    return grad



def damped_N(grad, Hess, x0, n, n_a, n_b, mu=1):
    '''Evaluate 1 damped newton step, in particular:
    if delta less then 1 return x0 and delta
    else perform a damped newton step and return new x and 1'''

    # Define symbols
    mu_ = symbols("mu")
    h, c, p, s, t = Define_symbols(n, n_a, n_b)

    # Make the gradient a callable function
    num_grad = lambdify((mu_, h, c, p, s, t), grad, 'numpy')
    # Make Hessian a callable function
    num_Hess = lambdify((mu_, h, c, p, s, t), Hess, 'numpy')

    # Extract variables
    h = x0[:n]
    c = x0[n]
    p = x0[n + 1]
    s = x0[n+2:n + 2 + n_a]
    t = x0[n+2+n_a:]

    # Evaluate gradient at x0
    G = np.array(num_grad(mu, h, c, p, s, t))

    # Evaluate Hessian at x0
    H = np.array(num_Hess(mu, h, c, p, s, t))
    print(np.linalg.eigvals(H))
    # Evaluate newton step
    n = - np.linalg.solve(H, G)

    # Evaluate delta
    #    print(G)
    #    print(np.dot(G, -n))
    delta = np.sqrt(np.dot(G, -n))

    # Break if delta is less then 1
    if delta < 1:
        return x0, delta

    # Evaluate new x
    x = x0 + n/(1+delta)

    # Attentiom: returned delta is the one associated to x0, not to x
    return x, delta



# Define gradient of F
gradF = Differenciate(F, n, n_a, n_b)
'''for i in range(len(gradF)):
    print(gradF[i])'''
# Make the gradient a callable function
num_gradF = lambdify((mu, h, c, p, s, t), gradF, 'numpy')

#print(len(num_gradF(1, [0 for i in range(n)], 0, 5, [2 for i in range(n_a)], [2 for i in range(n_b)])))



# Define the Hessian of F
HessF = []
for i in range(len(gradF)):
    new_row = Differenciate(gradF[i], n, n_a, n_b)
    HessF += [new_row]
'''for i in range(len(HessF)):
    for j in range(len(HessF)):
        print(HessF[i,j])'''

# Make Hessian a callable function
num_HessF = lambdify((mu, h, c, p, s, t), HessF, 'numpy')

# Define feasible x0 and try to compute a damped newton step
x0 = [0 for i in range(n)] + [0] + [5] + [2 for i in range(n_a)] + [2 for i in range(n_b)]

'''
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
# Extract variables
h = x0[:n]
c = x0[n]
p = x0[n + 1]
s = x0[n + 2:n + 2 + n_a]
t = x0[n + 2 + n_a:]
# Initial mu
mu = 3
print(f'F = {num_F(mu, h, c, p, s, t)}')
delta = 1
while delta >= 1:
    x0, delta = damped_N(gradF, HessF, x0, n, n_a, n_b, mu)
    # Extract variables
    h = x0[:n]
    c = x0[n]
    p = x0[n + 1]
    s = x0[n + 2:n + 2 + n_a]
    t = x0[n + 2 + n_a:]

    print(f'F = {num_F(mu, h, c, p, s, t)}')
    print(f'delta = {delta}')

''' Testing...
H = num_HessF(1, [0 for i in range(n)], 0, 5, [2 for i in range(n_a)], [2 for i in range(n_b)])
print(H)
print(np.array(H).shape)'''


