import numpy as np
import matplotlib.pyplot as plt

def dataset4():
    n = 20
    x_d4 = 3 * (np.random.rand(n, 4) - 0.5)
    y_d4 = (2 * x_d4[:, 0] - 1 * x_d4[:, 1] + 0.5 + 0.5 * np.random.randn(n)) > 0
    y_d4 = 2 * y_d4 - 1
    return x_d4, y_d4

def calc_gram_matrix(x, y):
    n = x.shape[0]
    d = x.shape[1]
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            res[i, j] = y[i] * y[j] * (x[i].T @ x[j])
    return res

def projection(inp):
    res = np.zeros(inp.shape)
    for i in range(inp.shape[0]):
        if 0<=inp[i][0] <= 1:
            res[i][0] = inp[i][0]
        elif inp[i][0] > 1:
            res[i][0] = 1
        else:
            res[i][0] = 0
    return res

def calc_w(x, y, alpha, lamb):
    n = x.shape[0]
    d = x.shape[1]
    res = np.zeros(d)
    for i in range(n):
        res += alpha[i] * y[i] * x[i]
    return res / 2 / lamb

def calc_hinge_loss(x, y, w, lamb):
    res = 0.0
    n = x.shape[0]
    for i in range(n):
        res += max(0, 1 - y[i] * (w.T @ x[i]))
    res += lamb * np.linalg.norm(w, ord=2) ** 2
    return res

def calc_dual_hinge_loss(alpha, gram, lamb):
    one = np.ones((alpha.shape[0], 1))
    res = - 1/(4 * lamb) * alpha.T @ gram @ alpha + alpha.T @ one
    return res[0,0]

def main():
    x, y = dataset4()
    n = x.shape[0]
    d = x.shape[1]
    w = np.random.rand(d)
    alpha = np.random.randn(n,1)
    alphas = [alpha]
    dual_hinge_losses = []
    hinge_losses = []
    lamb = 0.01
    for t in range(300):
        gram_matrix = calc_gram_matrix(x, y)
        lr = 0.001 / np.sqrt(t+1)
        alpha = projection(alpha - lr * (1/(2 * lamb) * gram_matrix @ alpha - 1))
        alphas.append(alpha)
        w = calc_w(x, y, alpha, lamb)
        dual_hinge_loss = calc_dual_hinge_loss(alpha, gram_matrix, lamb)
        hinge_loss = calc_hinge_loss(x, y, w, lamb)
        dual_hinge_losses.append(dual_hinge_loss)
        hinge_losses.append(hinge_loss)
    plt.plot(dual_hinge_losses, label='dual loss')
    plt.plot(hinge_losses, label='hinge loss')
    plt.xlabel('step')
    plt.ylabel('loss value')
    plt.legend()
    plt.savefig('problem_3.png')

if __name__ == '__main__':
    main()