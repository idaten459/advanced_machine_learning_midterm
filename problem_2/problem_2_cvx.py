import cvxpy as cv
import numpy as np

np.random.seed(10)

def opt(lambda_):
    A = np.array([[3,0.5],[0.5,1]])
    mu = np.array([[1,2]]).T
    w = cv.Variable(mu.shape)
    objective = cv.Minimize(cv.quad_form(w-mu, A) + lambda_ * cv.norm(w,1))
    prob = cv.Problem(objective)
    result = prob.solve()
    return w.value

def main():
    lambdas = [2,4,6]
    for lambda_ in lambdas:
        print(opt(lambda_))

if __name__ == '__main__':
    main()