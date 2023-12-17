import numpy as np
import Fun_Jac_Hess as fun


def Read_Data(n, n_a, n_b):
    image_size = 28  # width and length
    no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    data_path = "data/mnist/"
    train_data = np.loadtxt(data_path + "mnist_train.csv",
                            delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv",
                           delimiter=",")

    # Extract 0s from the training set
    # print(train_data[0])
    A = train_data[train_data[:, 0] == 0.]

    # Delete the first column
    A = A[0:n_a, 1:]

    # Only keep some columns such that first element is non zero (for start 10 of them)
    # non_zero_ind = np.nonzero(A[0, :])[0]
    # indices = np.random.choice(non_zero_ind, size=n, replace=False)
    # A = A[:, indices]

    # ***************************Old B
    # Extract non-0s
    # B = train_data[train_data[:, 0] != 0.]
    # Delete first column
    # B = B[0:n_b, 1:]
    # Keep the same columns you kept in A
    # B = B[:, indices]

    # ***************************New B
    B_data = np.zeros((int(n_b), len(train_data[0])))
    count = np.ones(10) * n_b / 9  # count the number of digit i at index [i]
    totalcount = n_b  # count the lines in B
    fac = 0.99 / 255
    for line in train_data:
        if line[0] != 0.:
            if count[int(line[0])] != 0:
                B_data[int(n_b - totalcount)] = line
                count[int(line[0])] -= 1
                totalcount -= 1

    # Normalize A and B [0.1, 1]
    A_fin = np.asfarray(A[:, :]) * fac + 0.01
    B_fin = np.asfarray(B_data[:, 1:]) * fac + 0.01

    return A_fin, B_fin


def short_path_method(A, B, lambd=5, eps=1e-3):
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

    # Initialize with x_0 and mu_0
    x_0, mu_0, _ = initialization(A, B, lambd)

    # Choose tau
    tau = .25

    # Choose theta v
    v = 2 * n_a + 2 * n_b + 1
    theta = (16 * np.sqrt(v)) ** (-1)

    # Compute mu_final
    mu_f = eps * (1 - tau) / v

    mu = mu_0
    x = x_0
    # While cycle
    while mu > mu_f:
        # Update mu
        mu = (1 - theta) * mu

        # Evaluate Gradient
        J = fun.Jacobian(x, mu, lambd, A, B)

        # Evalutate Hessian
        H = fun.Hessian(x, lambd, A, B)

        # Evaluate newton step
        newton = np.linalg.solve(H, J)
        print(f'Norma n_mu = {np.linalg.norm(newton)}')

        # Evaluate new x
        x = x - newton

    return x


def initialization(A, B, lambd):
    '''First draft: chose an interior point x_0 and find a mu_0 such that
    delta_mu_0 < 1'''

    # Number of features
    n = len(A[0, :])
    # Number of points in A
    n_a = len(A)
    # Number of points in B
    n_b = len(B)

    # Choose mu_0
    mu_0 = 1

    x0 = np.array([0 for i in range(n)] + [0] + [5] + [2 for i in range(n_a)] + [2 for i in range(n_b)])

    # Initialize delta
    delta = 1
    # Get closer to path until delta < 1
    while delta >= 1:
        # Do a Damped Newton step to decrease delta
        x0, delta = damped_N(x0, A, B, lambd, mu_0)
        print(f'delta = {delta}')
    return x0, mu_0, delta


def damped_N(x0, A, B, lambd, mu=1):
    '''Evaluate 1 damped newton step, in particular:
    if delta less then 1 return x0 and delta
    else perform a damped newton step and return new x and 1'''

    # Evaluate gradient at x0
    J = fun.Jacobian(x0, mu, lambd, A, B)

    # Evaluate Hessian at x0
    H = fun.Hessian(x0, lambd, A, B)

    # Evaluate newton step
    newton = np.linalg.solve(H, J)

    # Evaluate delta
    #    print(G)
    #    print(np.dot(G, n))
    delta = np.sqrt(np.dot(J, newton))

    # Break if delta is less then 1
    if delta < 1:
        return x0, delta

    # Evaluate new x
    x = x0 - newton / (1 + delta)

    # Attentiom: returned delta is the one associated to x0, not to x
    return x, delta
def update_x0(A,B,lambd):
    with open('x0.txt', 'wb') as fileX0:
        with open('mu0.txt', 'wb') as fileMu:
            with open('delta.txt', 'wb') as fileDelta:
                x0, mu_0, delta  = initialization(A, B, lambd=5)
                np.save(fileX0, x0)
                np.save(fileMu, mu_0)
                np.save(fileDeltadelta)

def load_x0():
    with open('x0.txt', 'rb') as fileX0:
        with open('mu0.txt', 'rb') as fileMu:
            with open('delta.txt', 'rb') as fileMu:
                x0 = np.load(fileX0)
                mu_0 = np.load(fileMu)
                delta = np.load(fileDelta)
    return x0, mu_0, delta

nine = 9
A, B = Read_Data(1*nine, 1*nine, 1*nine)
update_x0(A, B, lambd=5)
x0, mu_0, delta = load_x0()
print(f'delta = {delta}')

