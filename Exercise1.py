import numpy as np
from matplotlib import pyplot as plt
import math
# FIRST ========================================
def firstEquation(X):
    x, y = X
    return x**4 + y**4 - 4*x*y + 0.5*y + 1

def firstEquationDeri(X):
    x, y = X
    return np.array(4*x**3 - 4*y, -4*x + 4*y**3 + 0.5)

# SECOND =======================================
def secondEquation(X):
    x1, x2 = X
    return (100 * (x2-(x1**2))**2) + (1 - x1)**2

def secondEquationDeri(X):
    x1, x2 = X
    return np.array(2*(200*x1**3 - 200*x1*x2 + x1 - 1 ), 200*(x2-x1**2))

# THIRD =======================================
def thirdEquation(x):
    '''
    X is an array form x1 to x10
    '''
    return sum([
        (100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2)
        for i in range(1, 9)
    ])

def thirdEquationDeri(X):
    gradient = []

    gradient.append(secondEquationDeri([X[0], X[1]]))
    for i in range(1, 9):
        x, y, z = X[i-1], X[i], X[i+1]
        val = 200*(z - y**2)*(-2*y) - 2*(1 - y) + 200*(y - x**2)
        gradient.append(val)
    gradient.append(200*(X[9] - X[8]**2))

    return np.array(gradient)

def stepDepeest(x0, f, df, alpha= 0.1, maxI=500, tolx=1e-4, tolf= 1e-4, tolg=1e-4, metric = 2):
    iters = 0
    fin = 0
    convergence = 0
    x = x0.copy()

    approx = []
    values = []
    grads = []
    errs = []
    approx.append(x)
    values.append(f(x0))
    grads.append(df(x0))
    errs.append(np.linalg.norm(x0, 1))

    while(fin == 0 and iters < maxI):
        oldx = x
        oldalpha = alpha
        gr = df(oldx)
        x = oldx - alpha * gr
        fx = f(oldx)
        dfx = df(x)

        iters += 1
        approx.append(x)
        values.append(fx)
        grads.append(dfx)

        error = np.linalg.norm(oldx-x, metric)
        if error < tolx:
            fin = 1
            convergence = 1
        if (math.isnan(error) != False):
            errs.append(error)

    print("=======================================")
    print("Best: ", values[-1])
    print("Errors: ", errs)
    print("Point vals: ", approx)
    print("Func vals: ", values)
    print("Iters: ", iters)
    print("Convergent: ", convergence)
    print("=======================================")
    return [values, errs, approx, values, iters, convergence]

if __name__ == '__main__':
    errs1 = stepDepeest(np.array([0., 1.]), firstEquation, firstEquationDeri)[1]
    errs2 = stepDepeest(np.array([1., 3.]), secondEquation, secondEquationDeri)[1]
    errs3 = stepDepeest(np.array([1., 0., 0., 1., 1., 3., 0., 1., 0., 0.]), thirdEquation, thirdEquationDeri)[1]

    for err in (errs1, errs2, errs3):
        plt.figure()
        plt.plot(err[1:len(err)])
        plt.xlabel('Step')
        plt.ylabel('Error')
        plt.show()

