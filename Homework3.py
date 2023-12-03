import numpy as np
import time
import matplotlib.pyplot as plt

def interior_point_method(A, B, lambd):
    # Initialize primal and dual variables
    h = np.zeros((len(A[0]), 1))
    c = 0
    s = np.ones((len(A), 1))
    t = np.ones((len(B), 1))

    # Parameters
    mu = 10  # Barrier update factor
    epsilon = 1e-6  # Convergence tolerance
    max_iter = 100  # Maximum iterations

    for k in range(max_iter):
        # Construct the barrier objective and constraints
        obj = lambd * np.linalg.norm(h)**2 - np.sum(np.log(s)) / len(A) - np.sum(np.log(t)) / len(B)
        constraints = np.concatenate([h.T @ A[i] + c + s[i] - 1 for i in range(len(A))] +
                                      [-(h.T @ B[i] + c) + t[i] - 1 for i in range(len(B))])

        # Check convergence
        if np.linalg.norm(constraints) < epsilon:
            break

        # Construct the Jacobian matrix
        J = np.concatenate([np.hstack([2 * lambd * h.T, np.zeros((1, len(A[0]))), np.eye(len(A)) / s[i], np.zeros((1, len(B)))])
                            for i in range(len(A))] +
                           [np.hstack([np.zeros((1, len(A[0]))), np.zeros((1, 1)), np.zeros((1, len(A))), -np.eye(len(B)) / t[i]])
                            for i in range(len(B))])

        # Solve the linear system using Newton's method
        delta = np.linalg.solve(J, constraints)

        # Update variables
        h -= delta[:len(h)].reshape(h.shape)
        c -= delta[len(h)]
        s -= delta[len(h) + 1:len(h) + 1 + len(A)].reshape(s.shape)
        t -= delta[len(h) + 1 + len(A):].reshape(t.shape)

        # Update barrier parameters
        s = mu * s
        t = mu * t

    return h, c, obj


def long_step_interior_point_method(A, B, lambd):
    # Initialize primal and dual variables
    h = np.zeros((len(A[0]), 1))
    c = 0
    s = np.ones((len(A), 1))
    t = np.ones((len(B), 1))

    # Parameters
    mu = 10  # Barrier update factor
    epsilon = 1e-6  # Convergence tolerance
    max_iter = 100  # Maximum iterations

    for k in range(max_iter):
        # Construct the barrier objective and constraints
        obj = lambd * np.linalg.norm(h) ** 2 - np.sum(np.log(s)) / len(A) - np.sum(np.log(t)) / len(B)
        constraints = np.concatenate([h.T @ A[i] + c + s[i] - 1 for i in range(len(A))] +
                                     [-(h.T @ B[i] + c) - t[i] + 1 for i in range(len(B))])

        # Check convergence
        if np.linalg.norm(constraints) < epsilon:
            break

        # Construct the Jacobian matrix
        J = np.concatenate(
            [np.hstack([2 * lambd * h.T, np.zeros((1, len(A[0]))), np.eye(len(A)) / s[i], np.zeros((1, len(B)))])
             for i in range(len(A))] +
            [np.hstack([np.zeros((1, len(A[0]))), np.zeros((1, 1)), np.zeros((1, len(A))), -np.eye(len(B)) / t[i]])
             for i in range(len(B))])

        # Solve the linear system using Newton's method
        delta = np.linalg.solve(J, constraints)

        # Update variables
        h -= delta[:len(h)].reshape(h.shape)
        c -= delta[len(h)]
        s -= delta[len(h) + 1:len(h) + 1 + len(A)].reshape(s.shape)
        t -= delta[len(h) + 1 + len(A):].reshape(t.shape)

        # Update barrier parameters
        s = mu * s
        t = mu * t

    return h, c, obj


def interior_try(A, B, lambd):
    return 0


image_size = 28  # width and length
no_of_different_labels = 10  #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
# data_path = "data/mnist/"
train_data = np.loadtxt("mnist_train.csv",
                        delimiter=",")
test_data = np.loadtxt("mnist_test.csv",
                       delimiter=",")


fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
#train_imgs2 = np.asfarray(train_data) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
count = np.zeros(10)
for line in train_data:
    count[int(line[0])] +=1
print(count)
min = np.min(count[1:])
if min*9 > count[0]:
    min = count[0]//9
print(min)
count = np.ones(10)*min
count[0] *= 9
totalcount=9*min
print(count)
A_data = np.zeros((int(min*9), len(train_data[0])))
B_data = np.zeros((int(min*9), len(train_data[0])))
for line in train_data:
    if line[0] == 0.:
        if count[0] != 0:
            A_data[int(9*min-count[0])] = line
            count[0] -= 1
    else:
        if count[int(line[0])] != 0:
            B_data[int(9*min-totalcount)] = line
            count[int(line[0])] -= 1
            totalcount -= 1
print(A_data)
A_imgs = np.asfarray(A_data[:, 1:]) * fac + 0.01
B_imgs = np.asfarray(B_data[:, 1:]) * fac + 0.01

print(f"train img{train_imgs}")
#print(f"train img{train_imgs2}")
print(f"test img {test_imgs}")
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])
print(f"labels A{train_labels}")
print(f"labels B{test_labels}")

lr = np.arange(no_of_different_labels)
print(f"lr :{lr}")

# transform labels into one hot representation
train_labels_one_hot = (lr == train_labels).astype(float)
test_labels_one_hot = (lr == test_labels).astype(float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot == 0] = 0.01
train_labels_one_hot[train_labels_one_hot == 1] = 0.99
test_labels_one_hot[test_labels_one_hot == 0] = 0.01
test_labels_one_hot[test_labels_one_hot == 1] = 0.99


for i in range(10):
    img = A_imgs[i].reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.show()
for i in range(10):
    img = B_imgs[i].reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.show()

A = train_labels_one_hot
B = test_labels_one_hot
print(f"A: {A}")
print(f"B: {B}")
# Assuming you have loaded the MNIST data into A and B
# A contains zero digits, and B contains the other digits

# Set the regularization parameter lambda
lambd = 0.1

start_time = time.time()

# Run the interior point method
h, c, obj = interior_point_method(A_imgs, B_imgs, lambd)

# Output the results
print(f"Optimal h: {h}")
print(f"Optimal c: {c}")
print(f"Optimal obj: {obj}")
print(f"Total CPU time: {time.time() - start_time} seconds")
