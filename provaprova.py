import numpy as np
from scipy.optimize import minimize
from sympy import symbols, lambdify

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



def Barrier(x,lambd,A,B):
    '''A,B are the matrices defining parameters of G
    x is an 1D-array of lenght n+n_a+n_b+2, x = [h,c,p,s,t]'''

    # Define the indices
    # Number of features
    n = len(A[0, :])
    # Number of points in A
    n_a = len(A)
    # Number of points in B
    n_b = len(B)

    # Extract variables
    h = x0[:n]
    c = x0[n]
    p = x0[n + 1]
    s = x0[n + 2:n + 2 + n_a]
    t = x0[n + 2 + n_a:]

    # Initialize the value of barrier at 0
    G = 0

    for i in range(n_a):
        G -= np.log(s[i])
        G -= np.log(-1+s[i]-np.dot(h,A[i,:])-c)

    for i in range(n_b):
        G -= np.log(t[i])
        G -= np.log(np.dot(h,B[i,:]) + c -1 + t[i])

    obj = Old_Objective(lambd,h,s,t)

    G -= np.log(p - obj)

    return G

def Old_Objective(lambd,h,s,t):
    '''Evaluates the objective of the SVM problem in the classical formulation'''

    obj = lambd * np.dot(h,h) + np.mean(s) + np.mean(t)

    return obj

def New_Obj(x,mu,lambd,A,B):
    '''Evaluates the objective of the SVM problem in the formulation
    linear + barrier'''

    # Extract variables
    p = x0[n + 1]

    # Evaluate Barrier
    Bar = Barrier(x,lambd,A,B)

    # Evaluate Objective
    val = p/mu + Bar

    return val

# Number of features
n = len(A[0, :])
# Number of points in A
n_a = len(A)
# Number of points in B
n_b = len(B)

# Define feasible x0 and try to compute a damped newton step
x0 = np.array([0 for i in range(n)] + [0] + [5] + [2 for i in range(n_a)] + [2 for i in range(n_b)])
print(f'x0 = {x0}')


# Define mu
mu = 1

# Define parameter lambd
lambd = 5

Val = New_Obj(x0,mu,lambd,A,B)
print(f'F = {Val}')

# Call minimize
result = minimize(fun=lambda x: New_Obj(x, mu, lambd, A, B), x0=x0, method='L-BFGS-B')

# Extract the result
optimized_x = result.x
optimized_value = result.fun

print("Optimized x:", optimized_x)
print("Optimized value:", optimized_value)

