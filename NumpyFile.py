import numpy as np
import pandas as pd
import Fun_Jac_Hess_v2 as fun
import SPFM as SPFM
from datetime import datetime
#from scratch import update_short_hctime


def MakeNumpyFileLong():
    lambdaList = [1, 5, 10]
    nDigitList = [1, 2, 3]
    hList = []
    cList = []
    timeList = []
    nine=9

    #SPFM.update_x0(A,B)
    #x = SPFM.short_path_method(A, B, lambd = 5, eps = 1e-3)

    for lambd in lambdaList:

        for NumberDigit in nDigitList:
            A, B = SPFM.Read_Data(NumberDigit * nine, NumberDigit * nine)
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

def MakeNumpyFileShort():
    nine = 9
    lambdaList = [1, 5, 10]
    nDigitList = [1, 2]
    hList = []
    cList = []
    timeList = []
    #SPFM.update_x0(A,B)
    #x = SPFM.short_path_method(A, B, lambd = 5, eps = 1e-3)
    for lambd in lambdaList:

        for NumberDigit in nDigitList:
            A, B = SPFM.Read_Data(NumberDigit * nine, NumberDigit * nine)
            x0init, mu0init, deltainit, timeInit = SPFM.update_x0(A, B, lambd)
            with open(f'init_short_{NumberDigit}digit_{lambd}lambda.txt', 'w') as file:
                file.write("NDigit;x0;mu0;delta;time\n")
                file.write(f"{NumberDigit};{x0init};{mu0init};{deltainit};{timeInit}\n")

            exponentList = [2e1, 1e1]
            with open(f'solutions_short_{NumberDigit}digit_{lambd}lambda.txt', 'w') as file:
                file.write("eps;solution;time\n")
                for epsilon in exponentList:
                    x0init, mu0init, deltainit, timeInit = SPFM.update_x0(A, B, lambd)
                    x, time = SPFM.short_path_method(A, B, lambd=lambd, eps=epsilon)
                    n = len(A[0, :])
                    n_a = len(A)
                    h = x[:n]

                    #print(h)

                    if np.any(hList):
                        hList = np.concatenate((hList, [h]))
                    else:
                        hList = [h]
                    #print(hList)
                    c = x[n]
                    if np.any(cList):
                        cList = np.concatenate((cList, [c]))
                    else:
                        cList = [c]
                    #print(cList)
                    #p = x[n + 1]
                    #s = x[n + 2:n + 2 + n_a]
                    #t = x[n + 2 + n_a:]
                    if np.any(timeList):
                        timeList = np.concatenate((timeList, [time]))
                    else:
                        timeList = [time]

                    print(f'solutions_short_{NumberDigit}digit_{lambd}lambda.txt')
                    print(time)
                    file.write(f"{epsilon};{h};{c};{time}\n")
    print(hList)
    print(cList)
    update_short_hctime(hList, cList, timeList)



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

def update_short_hctime(h,c,time):
    with open('hshort.txt', 'wb') as fileH:
        with open('cshort.txt', 'wb') as fileC:
            with open('timeshort.txt', 'wb') as fileTime:
                np.save(fileH, h)
                np.save(fileC, c)
                np.save(fileTime, time)

def load_short_hctime():
    with open('hshort.txt', 'rb') as fileH:
        with open('cshort.txt', 'rb') as fileC:
            with open('timeshort.txt', 'rb') as fileTime:
                h = np.load(fileH)
                c = np.load(fileC)
                time = np.load(fileTime)
    return h, c, time