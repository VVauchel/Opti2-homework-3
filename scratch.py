import numpy as np
import Fun_Jac_Hess_v2 as fun
import prova as pr
from sympy import symbols, lambdify, log, diff
import SPFM as SPFM
from time import perf_counter
import pandas as pd
import ast

nine = 9
#lambd = 5

#NumberDigit = 1


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

def update_hctime(h,c,time):
    with open('h.txt', 'wb') as fileH:
        with open('c.txt', 'wb') as fileC:
            with open('time.txt', 'wb') as fileTime:
                np.save(fileH, h)
                np.save(fileC, c)
                np.save(fileTime, time)

def load_hctime():
    with open('h.txt', 'rb') as fileH:
        with open('c.txt', 'rb') as fileC:
            with open('time.txt', 'rb') as fileTime:
                h = np.load(fileH)
                c = np.load(fileC)
                time = np.load(fileTime)
    return h, c, time

def classifier(h,c,B):
    '''Let's have the classifier take in input h and c, then we can iterate outside of this function'''

    #cList, hList, timeList = load_hctime()



    TrueCount = 0
    FalseCount = 0
    # Classify all the testing set
    for j in range(len(B)):

        # Get the datapoint
        x = B[j][1:]
        # Classify
        if (np.sign(np.dot(h, x) + c) < 0 and B[j][0] == 0) or (
                np.sign(np.dot(h, x) + c) > 0 and B[j][0] > 0):
            TrueCount += 1
        else:
            FalseCount += 1


    return [TrueCount, FalseCount]

#hList, cList, timeList=load_hctime()

# Replace 'your_saved_file.csv' with the actual filename you used
filename = 'results_20231220192908.csv'

# Read the CSV file into a DataFrame
result_df = pd.read_csv(filename)

for i in range(result_df.shape[0]):

    h_string = result_df['h'][i]
    # Convert the string to a NumPy array
    h = np.fromstring(h_string.replace('[', '').replace(']', ''), sep=' ')
    c = result_df['c'][i]
    a = classifier(h,c)[0]
    print(a)

hList, cList, timeList=load_hctime()
B = SPFM.Read_Data_Test(0)
for i in range(len(hList)):
    #print(hList[i])
    #print(cList[i])
    a = classifier(hList[i], cList[i], B)
    print(a)
print(timeList)
# Initial mu
mu = 1
#print(f'F = {num_F(mu, h, c, p, s, t)}')

# Initialize algorithm

#x0, mu_0, delta = SPFM.initialization(A, B, lambd=5)
#print(f'x0 = {x0[:10]}, delta = {delta}')
#NumberDigit=1
#A, B = SPFM.Read_Data(NumberDigit * nine, NumberDigit * nine, NumberDigit * nine)
#x0init, mu0init, deltainit, timeInit = SPFM.update_x0(A, B, 5)
#x = SPFM.long_path_method(A, B, lambd = 5, eps = 1e-3)

"""
lambdaList = [1, 5, 10]
nDigitList = [1, 2, 3]
hList = []
cList = []
timeList = []

#SPFM.update_x0(A,B)
#x = SPFM.short_path_method(A, B, lambd = 5, eps = 1e-3)



for lambd in lambdaList:

    for NumberDigit in nDigitList:
        A, B = SPFM.Read_Data(NumberDigit * nine, NumberDigit * nine, NumberDigit * nine)
        x0init, mu0init, deltainit, timeInit = SPFM.update_x0(A, B, lambd)
        with open(f'init_long_{NumberDigit}digit_{lambd}lambda.txt', 'w') as file:
            file.write("NDigit;x0;mu0;delta;time\n")
            file.write(f"{NumberDigit};{x0init};{mu0init};{deltainit};{timeInit}\n")

        exponentList = [1e1, 5e0]
        with open(f'solutions_long_{NumberDigit}digit_{lambd}lambda.txt', 'w') as file:
            file.write("eps;solution;time\n")
            for epsilon in exponentList:
                x, time = SPFM.long_path_method(A, B, lambd=lambd, eps=epsilon)
                n = len(A[0, :])
                n_a = len(A)
                h = x[:n]

                print(h)

                if np.any(hList):
                    hList = np.concatenate((hList, [h]))
                else:
                    hList = [h]
                print(hList)
                c = x[n]
                if np.any(cList):
                    cList = np.concatenate((cList, [c]))
                else:
                    cList = [c]
                print(cList)
                #p = x[n + 1]
                #s = x[n + 2:n + 2 + n_a]
                #t = x[n + 2 + n_a:]
                if np.any(timeList):
                    timeList = np.concatenate((timeList, [time]))
                else:
                    timeList = [time]

                print(timeList)
                file.write(f"{epsilon};{h};{c};{time}\n")
print(hList)
print(cList)
update_hctime(hList, cList, timeList)
"""

"""for lambd in lambdaList:
    for NumberDigit in nDigitList:
        A, B = SPFM.Read_Data(NumberDigit * nine, NumberDigit * nine, NumberDigit * nine)
        x0init, mu0init, deltainit, timeInit = SPFM.update_x0(A, B, lambd)
        with open(f'init_{NumberDigit}digit_{lambd}lambda.txt', 'w') as file:
            file.write("NDigit;x0;mu0;delta;time\n")
            file.write(f"{NumberDigit};{x0init};{mu0init};{deltainit};{timeInit}\n")

        exponentList = [1e1, 5e0]
        with open(f'solutions_{NumberDigit}digit_{lambd}lambda.txt', 'w') as file:
            file.write("eps;solution;time\n")
            for epsilon in exponentList:
                x, time = SPFM.short_path_method(A, B, lambd=lambd, eps=epsilon)
                file.write(f"{epsilon};{x};{time}\n")
"""


