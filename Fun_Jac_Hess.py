import numpy as np
def Barrier(x, lambd, A, B):
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
    h = x[:n]
    c = x[n]
    p = x[n + 1]
    s = x[n + 2:n + 2 + n_a]
    t = x[n + 2 + n_a:]

    # Initialize the value of barrier at 0
    G = 0

    for i in range(n_a):
        G -= np.log(s[i])
        G -= np.log(-1+s[i]-np.dot(h, A[i, :])-c)

    for i in range(n_b):
        G -= np.log(t[i])
        G -= np.log(np.dot(h, B[i, :]) + c -1 + t[i])

    obj = Old_Objective(lambd, h, s, t)

    G -= np.log(p - obj)

    return G

def Old_Objective(lambd, h, s, t):
    '''Evaluates the objective of the SVM problem in the classical formulation'''

    obj = lambd * np.dot(h, h) + np.mean(s) + np.mean(t)

    return obj

def New_Obj(x, mu, lambd, A, B):
    '''Evaluates the objective of the SVM problem in the formulation
    linear + barrier'''

    # Number of features
    n = len(A[0, :])
    # Number of points in A
    n_a = len(A)
    # Number of points in B
    n_b = len(B)

    # Extract variables
    p = x[n + 1]

    # Evaluate Barrier
    Bar = Barrier(x, lambd, A, B)

    # Evaluate Objective
    val = p/mu + Bar

    return val

def Jacobian(x, mu, lambd, A, B):
    '''Evaluates the jacobian of the new objective in x'''

    # Number of features
    n = len(A[0, :])
    # Number of points in A
    n_a = len(A)
    # Number of points in B
    n_b = len(B)

    # Extract variables
    h = x[:n]
    c = x[n]
    p = x[n + 1]
    s = x[n + 2:n + 2 + n_a]
    t = x[n + 2 + n_a:]


    # Three recurrent quantities in the expressions of derivatives...
    Q1 = [-1 + s[i] - np.dot(h, A[i, :]) - c for i in range(n_a)]
    Q2 = [np.dot(h, B[i, :]) + c - 1 + t[i] for i in range(n_b)]
    Q3 = p - Old_Objective(lambd, h, s, t)

    #dF/dh_j
    Jh = [sum(A[i, j]/Q1[i] for i in range(n_a))
         - sum(B[i, j]/Q2[i] for i in range(n_b))
         + 2*lambd*h[j]/Q3 for j in range(n)]

    #dF/dc
    Jc = [sum(1/Q1[i] for i in range(n_a)) - sum(1/Q2[i] for i in range(n_b))]

    #dF/dp
    Jp = [1/mu - 1/Q3]

    #dF/ds_j
    Js = [-1/s[j]
          - 1/Q1[j]
          + 1/(Q3*n_a)
          for j in range(n_a)]

    #dF/dt_j
    Jt = [-1/t[j]
          - 1/Q2[j]
          + 1/(Q3*n_b)
          for j in range(n_b)]

    J = Jh + Jc + Jp + Js + Jt

    J = np.array(J)
    return J

def Hessian(x, lambd, A, B):
    '''Evaluates the Hessian of the new_objective'''

    # Number of features
    n = len(A[0, :])
    # Number of points in A
    n_a = len(A)
    # Number of points in B
    n_b = len(B)

    # Extract variables
    h = x[:n]
    c = x[n]
    p = x[n + 1]
    s = x[n + 2:n + 2 + n_a]
    t = x[n + 2 + n_a:]

    # Three recurrent quantities in the expressions of derivatives...
    Q1 = [-1 + s[i] - np.dot(h, A[i, :]) - c for i in range(n_a)]
    Q2 = [np.dot(h, B[i, :]) + c - 1 + t[i] for i in range(n_b)]
    Q3 = p - Old_Objective(lambd, h, s, t)
    # ...but the first two squared
    Q1 = [Q1[i]**2 for i in range(n_a)]
    Q2 = [Q2[i]**2 for i in range(n_b)]

    # Initialize H
    H = [[0 for i in range(n+n_a+n_b+2)] for j in range(n+n_a+n_b+2)]

    # H1

    # Derive dF/dh_j for each j
    for j in range(n):
        # dF/dh_j dh_k
        Hhh = [sum(A[i, j]*A[i, k]/Q1[i] for i in range(n_a))
        + sum(B[i, j]*B[i, k]/Q2[i] for i in range(n_b))
        + (j == k)*2*lambd/Q3 + 4*lambd*h[j]*h[k]/(Q3**2)
        for k in range(n)]

        # dF/dh_j dc
        Hhc = [sum(A[i, j]/Q1[i] for i in range(n_a))
               + sum(B[i, j]/Q2[i] for i in range(n_b))]

        # dF/dh_j dp
        Hhp = [- 2*lambd*h[j]/(Q3**2)]

        # dF/dh_j ds_k
        Hhs = [- A[k, j]/Q1[k] + 2*lambd*h[j]/(n_a*(Q3**2)) for k in range(n_a)]

        # dF/dh_j dt_k
        Hht = [B[k, j]/Q2[k] + 2*lambd*h[j]/(n_b*(Q3**2)) for k in range(n_b)]

        # Define the j-th row of hessian
        H[j] = Hhh + Hhc + Hhp + Hhs + Hht

    # H2
    # dF/dc dh_k
    Hch = [sum(A[i, k]/Q1[i] for i in range(n_a))
           + sum(B[i, k]/Q2[i] for i in range(n_b))
           for k in range(n)] #This was already evaluated and could be changed

    # dF/dc dc
    Hcc = [sum(1/Q1[i] for i in range(n_a))
           + sum(1/Q2[i] for i in range(n_b))]

    # dF/dc dp
    Hcp = [0]

    # dF/dc ds_k
    Hcs = [- 1/Q1[k] for k in range(n_a)]

    # dF/dc dt_k
    Hct = [1/Q2[k] for k in range(n_b)]

    # Define the (n+1)-th row of H
    H[n] = Hch + Hcc + Hcp + Hcs + Hct

    # H3
    # dF/dp dh_k
    Hph = [H[k][n+1] for k in range(n)]

    # dF/dp dc
    Hpc = [0]

    # dF/dp dp
    Hpp = [1/(Q3**2)]

    # dF/dp ds_k
    Hps = [-Hpp[0]/n_a for k in range(n_a)]

    # dF/dp dt_k
    Hpt = [-Hpp[0]/n_b for k in range(n_b)]

    # Define the (n+2)-th row of H
    H[n+1] = Hph + Hpc + Hpp + Hps + Hpt

    # H4

    for j in range(n_a):
        # dF/ds_j dh_k
        Hsh = [H[k][n+2+j] for k in range(n)]

        # dF/ds_j dc
        Hsc = [H[n][n+2+j]]

        # dF/ds_j dp
        Hsp = [H[n+1][n+2+j]]

        # dF/ds_j ds_k
        Hss = [(j == k)*(1/s[j]**2 + 1/Q1[j])
              + 1/(Q3*n_a)**2
               for k in range(n_a)]

        # dF/ds_j dt_k
        Hst = [(Q3**2 * n_a * n_b)**(-1) for k in range(n_b)]

        H[n+2+j] = Hsh + Hsc + Hsp + Hss + Hst

    # H5
    for j in range(n_b):
        # dF/dt_j dh_k
        Hth = [H[k][n+2+n_a+j] for k in range(n)]

        # dF/dt_j dc
        Htc = [H[n][n+2+n_a+j]]

        # dF/dt_j dp
        Htp = [H[n+1][n+2+n_a+j]]

        # dF/dt_j ds_k
        Hts = [H[n+2+k][n+2+n_a+j] for k in range(n_a)]

        # dF/dt_j dt_k
        Htt = [(j == k)*(1/t[j]**2 + 1/Q2[j])
              + 1/(Q3*n_b)**2
               for k in range(n_b)]

        H[n + 2 + n_a + j] = Hth + Htc + Htp + Hts + Htt

    H = np.array(H)
    return H

'''
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

Jac = Jacobian(x0,mu,lambd,A,B)
print(f'Jac = {Jac}')

Hes = Hessian(x0,mu,lambd,A,B)
print(f'Hes = {Hes}')
'''


