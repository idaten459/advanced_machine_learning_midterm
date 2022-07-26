from re import S
import numpy as np
import matplotlib.pyplot as plt
import cvxpy

def dataset4():
    n = 10
    x_d4 = 3 * (np.random.rand(n, 4) - 0.5)
    y_d4 = (2 * x_d4[:, 0] - 1 * x_d4[:, 1] + 0.5 + 0.5 * np.random.randn(n)) > 0
    y_d4 = 2 * y_d4 - 1
    return x_d4, y_d4

# return sigmoid(x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calc_loss(x, y, w, lamb=0.01):
    res = 0.0
    n = x.shape[0]
    for i in range(n):
        res += np.log(1 + np.exp(-y[i] * (w.T @ x[i])))
    res += lamb * np.linalg.norm(w, ord=2) ** 2
    return res

def calc_grad(x, y, w, lamb=0.01):
    res = 0.0
    n = x.shape[0]
    for i in range(n):
        res += -y[i] * x[i] * np.exp(-y[i] * (w.T @ x[i])) / (1 + np.exp(y[i] * (w.T @ x[i])))
    res += 2 * lamb * w
    return res

def batch_steepest_gradient(epoch=100, lr=1.0):
    x_d4, y_d4 = dataset4()
    loss_hist_batch = []
    n = x_d4.shape[0]
    d = x_d4.shape[1]
    w = np.random.rand(d)
    for _ in range(epoch):
        grad = calc_grad(x_d4, y_d4, w)
        w -= lr * grad
        loss_hist_batch.append(calc_loss(x_d4, y_d4, w))
    return loss_hist_batch

def calc_hess(x, y, w, lamb=0.01):
    res = 0.0
    n = x.shape[0]
    d = x.shape[1]
    for i in range(n):
        #print(f'x[{i}]',x[i])
        #print(x[i] @ x[i].T)
        x_mat = np.array([x[i]]).T
        #print('x_mat',x_mat)
        #print(x_mat @ x_mat.T)
        res += np.exp(-y[i] * (w.T @ x[i])) / (1 + np.exp(-y[i] * (w.T @ x[i]))) ** 2 * (x_mat @ x_mat.T) * y[i] ** 2
    res += 2 * lamb * np.eye(d)
    return res

def newton_method(epoch=100, lr=1.0):
    x_d4, y_d4 = dataset4()
    #print(y_d4.shape)
    loss_hist_newton = []
    n = x_d4.shape[0]
    d = x_d4.shape[1]
    w = np.random.rand(d)
    for _ in range(epoch):
        grad = calc_grad(x_d4, y_d4, w)
        hess = calc_hess(x_d4, y_d4, w)
        #print(hess)
        w -= lr * np.linalg.inv(hess) @ grad
        loss_hist_newton.append(calc_loss(x_d4, y_d4, w))
        #print(_,calc_loss(x_d4, y_d4, w))
    return loss_hist_newton

if __name__ == '__main__':
    np.random.seed(42)
    loss_hist_batch = batch_steepest_gradient(epoch=200,lr=0.1)
    loss_hist_newton = newton_method(epoch=200,lr=0.1)
    show_iter = 200
    min_loss = min(np.min(loss_hist_batch), np.min(loss_hist_newton)) # ここ要修正かも
    plt.plot(np.abs(loss_hist_batch[:show_iter]-min_loss), label='steepest')
    plt.plot(np.abs(loss_hist_newton[:show_iter]-min_loss), label='newton')
    plt.legend()
    plt.yscale('log')
    plt.show()