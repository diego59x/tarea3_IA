import numpy as np


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
    # 2 (-1 + x + 200 x^3 - 200 x y)


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
    # print(X)
    # print(secondEquationDeri([X[0], X[1]]))
    gradient.append(secondEquationDeri([X[0], X[1]]))
    for i in range(1, 9):
        x, y, z = X[i-1], X[i], X[i+1]
        val = 200*(z - y**2)*(-2*y) - 2*(1 - y) + 200*(y - x**2)
        gradient.append(val)
    gradient.append(200*(X[9] - X[8]**2))

    return np.array(gradient)

# def f_rosenbrock(xy):
#     x, y = xy
#     return 100 * (y - x**2)**2 + (1 - x)**2

# def df_rosenbrock(xy):
#     x, y = xy
#     return np.array([-400*x*(y-x**2)-2*(1-x), 200*(y-x**2)])

def gradient_descent(f, df, x0, tol=.1, alpha=1.0, ratio=.8, c=.01):
    x_k, num_steps, step_size = x0, 0, alpha
    while True:
        g_k = df(x_k)

        if np.abs(g_k).max() < tol:
            break

        num_steps += 1

        fx, cg = f(x_k), - c * (g_k**2).sum()
        while f(x_k - step_size * g_k) > fx + step_size * cg:
            step_size *= ratio

        x_k -= step_size * g_k

    return x_k, g_k, num_steps

if __name__ == '__main__':
    x, g, n = gradient_descent(
        thirdEquation, thirdEquationDeri, np.array([0., 0., 0., 0.,0.,0., 0., 0., 0.,0.])
    )
    print("The number of steps is: ", n)
    print("The final step is:", x)
    print("The gradient is: ", g)


# if __name__ == "__main__":
#     print(stepDepeest(np.array([0.0, 0.0]), quad, dquad))
