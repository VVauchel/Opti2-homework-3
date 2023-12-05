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
A = train_data[train_data[:,0] == 0.]
# Delete the first column
A = A[0:5:,1:]
A = A[:,0:10]


# Extract non-0s
B = train_data[train_data[:,0] != 0.]
# Delete first column
B = B[0:5,1:]
B = B[:,0:10]


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
    n = len(A[0,:])

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
    F = Barrier(h, c, s, t, p)

    # Initialize with x_0 and mu_0
    x_0, mu_0 = initialization_sspf(F, n, n_a, n_b)

    # While cycle


    return 1

def Define_symbols(n,n_a,n_b):

    # Define the simbols:
    h = symbols('h1:%d' % (n + 1))  # h as an n-dimensional vector
    c, p = symbols('c p')  # c and p as scalars
    s = symbols('s1:%d' % (n_a + 1))  # s as an n_a-dimensional vector
    t = symbols('t1:%d' % (n_b + 1))  # t as an n_b-dimensional vector

    return h, c, p, s, t

def Barrier(lambd,n,n_a,n_b):
    '''Self concordant barrier'''

    # Define Symbols
    h, c, p, s, t = Define_symbols(n,n_a,n_b)


    # Define the self concordant barrier
    F = 0
    F = F - sum(log(si) for si in s)
    F = F - sum(log(ti) for ti in t)
    F = F - sum(log(-1 + s[i] - sum(h[j]*A[i,j] for j in range(n)) - c) for i in range(n_a))
    F = F - sum(log(sum(h[j]*B[i,j] for j in range(n)) + c - 1 + t[i]) for i in range(n_b))
    F = F - log(p - Objective(lambd,n,n_a,n_b))

    return F

def Objective(lambd,n,n_a,n_b):
    '''Defines the objective function of ex A3 as a sympy function'''

    # Define Symbols
    h, c, p, s, t = Define_symbols(n,n_a,n_b)

    G = 0
    G = G + sum((h[i]**2) for i in range(n))
    G = G + lambd*G
    G = G + sum(s[i] for i in range(len(s)))/len(s)
    G = G + sum(t[i] for i in range(len(t)))/len(t)
    return G

def initialization_sspf(F, gradF, HessF, n, n_a, n_b):
    '''First draft: chose an interior point x_0 and find a mu_0 such that 
    delta_mu_0 < 1'''




    mu_0 = 1
    x_0 = 1


    return x_0, mu_0

    n = len(A[0,:])

    # Number of points in A
    n_a = len(A)

    # Number of points in B
    n_b = len(B)


#start_time = time.time()

    # Number of features
n = len(A[0,:])

    # Number of points in A
n_a = len(A)

    # Number of points in B
n_b = len(B)

# Define the variables of the problem
mu = symbols("mu")
h, c, p, s, t = Define_symbols(n,n_a,n_b)

# Define the function to minimize at each step
F = p/mu + Barrier(lambd,n,n_a,n_b)

# Define it numerically
num_F = lambdify((h, c, p, s, t, mu), F, 'numpy')

# Save the function to a file
with open('my_function.pkl', 'wb') as file:
    dill.dump(num_F, file)

# Now, in a different script, you can load the function
with open('my_function.pkl', 'rb') as file:
    loaded_function = dill.load(file)

# Use the loaded function
print("Result using the loaded function:", loaded_function)

'''
gradF = [diff(F, p)]
gradF += [diff(F, h[i]) for i in range(n)]
gradF += [diff(F, c)]
gradF += [diff(F, s[i]) for i in range(n_a)]
gradF += [diff(F, t[i]) for i in range(n_b)]

num_gradF = lambdify((h, c, p, s, t), gradF, 'numpy')
'''



