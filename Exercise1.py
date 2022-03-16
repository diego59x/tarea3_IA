import numpy as np


# FIRST ========================================
def firstEquation(x, y):
    return x**4 + y**4 - 4*x*y + 0.5*y + 1

def firstEquationDeri(x, y):
    return np.array(4*x**3 - 4*y, -4*x + 4*y**3 + 0.5)


# SECOND =======================================
def secondEquation(x1, x2):
    return (100 * (x2-(x1**2))**2) + (1 - x1)**2

def secondEquationDeri(x1, x2):
    return np.array(2*(200*x1**3 - 200*x1*x2 + x1 - 1 ), 200*(x2-x1**2))
    # 2 (-1 + x + 200 x^3 - 200 x y)


# THIRD =======================================
def thirdEquation(x):
    '''
    X is an array form x1 to x10
    '''
    return sum([
        (100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2)
        for i in range(1, 10)
    ])

def thirdEquationDeri(x1, x2):
    ...


def stepDepeest(x0, f, df, alpha= 0.1, maxI=500, tolx=1e-4, tolf= 1e-4, tolg=1e-4, metric = 2, crit='abserrox'):
    iters = 0
    fin = 0
    convergence = 0
    x = x0.copy()
    n = x.shape[0]

    approx = []
    values = []
    grads = []
    errs = []
    approx.append(x)
    values.append(f(x0))
    grads.append(df(x0))
    errs.append(np.linalg.norm(x0, 1))

    while(fin == 0):
        oldx = x
        oldalpha = alpha
        gr = df(oldx)
        x = oldx - alpha * gr
        fx = f(x)
        dfx = df(x)

        iters += 1
        approx.append(x)
        values.append(fx)
        grads.append(dfx)

        if (crit == 'abserrox'):
            error = np.liang.norm(oldx-x, metric)
            if error < tolx:
                fin = 1
                convergence = 1
        elif (crit == 'relerrox'):
            error = np.linalg.norm(oldx-x, metric)/np.linalg.norm(x, metric)
            if (error < tolx):
                fin = 1
                convergence = 1


if __name__ == "__main__":
    print(stepDepeest(1, firstEquation, ))
