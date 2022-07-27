import numpy as np
import matplotlib.pyplot as plt
import time

def dataset5():
    n = 200
    x_d5 = 3 * (np.random.rand(n, 4) - 0.5)
    W = np.array([[ 2,  -1, 0.5,],
                [-3,   2,   1,],
                [ 1,   2,   3]])
    y_d5 = np.argmax(np.dot(np.hstack([x_d5[:,:2], np.ones((n, 1))]), W.T)
                            + 0.5 * np.random.randn(n, 3), axis=1)
    return x_d5, y_d5

def preprocess(x_d5, y_d5):
    num_classes = 3
    ty_d5 = np.eye(num_classes)[y_d5].T
    print(y_d5)
    print(ty_d5)
    return x_d5, ty_d5

def calc_loss(x, y, w, lamb=0.01):
    res = 0.0
    n = x.shape[0]
    for c in range(3):
        for i in range(n):
            res += np.log(1 + np.exp(-y[c][i] * (w[c].T @ x[i])))
        res += lamb * np.linalg.norm(w[c], ord=2) ** 2
    return res

def calc_grad(x, y, w, lamb=0.01):
    res = np.zeros(w.shape, dtype=np.float64)
    n = x.shape[0]
    for c in range(3):
        for i in range(n):
            res[c] += -y[c][i] * x[i] * np.exp(-y[c][i] * (w[c].T @ x[i])) / (1 + np.exp(-y[c][i] * (w[c].T @ x[i])))
        res[c] += 2 * lamb * w[c]
    return res

def batch_steepest_gradient(epoch=100, lr=1.0):
    x_d5, y_d5 = dataset5()
    x_d5, y_d5 = preprocess(x_d5, y_d5)
    loss_hist_batch = []
    n = x_d5.shape[0]
    d = x_d5.shape[1]
    w = np.random.rand(3,d)
    for _ in range(epoch):
        grad = calc_grad(x_d5, y_d5, w)
        w -= lr * grad
        loss_hist_batch.append(calc_loss(x_d5, y_d5, w))
    return loss_hist_batch

def calc_hess(x, y, w, lamb=0.01):
    res = np.zeros((3, w.shape[1], w.shape[1]), dtype=np.float64)
    n = x.shape[0]
    d = x.shape[1]
    for c in range(3):
        for i in range(n):
            #print(f'x[{i}]',x[i])
            #print(x[i] @ x[i].T)
            x_mat = np.array([x[i]]).T
            #print('x_mat',x_mat)
            #print((x_mat @ x_mat.T).shape)
            #print(res[c].shape)
            res[c] += np.exp(-y[c][i] * (w[c].T @ x[i])) / (1 + np.exp(-y[c][i] * (w[c].T @ x[i]))) ** 2 * (x_mat @ x_mat.T) * y[c][i] ** 2
        res[c] += 2 * lamb * np.eye(d)
    return res

def newton_method(epoch=100, lr=1.0):
    x_d5, y_d5 = dataset5()
    x_d5, y_d5 = preprocess(x_d5, y_d5)
    loss_hist_newton = []
    n = x_d5.shape[0]
    d = x_d5.shape[1]
    w = np.random.rand(3,d)
    for _ in range(epoch):
        grad = calc_grad(x_d5, y_d5, w)
        hess = calc_hess(x_d5, y_d5, w)
        #print(hess)
        for c in range(3):
            w[c] -= lr * np.linalg.inv(hess[c]) @ grad[c]
        loss_hist_newton.append(calc_loss(x_d5, y_d5, w))
        #print(_,calc_loss(x_d4, y_d4, w))
    return loss_hist_newton

if __name__ == '__main__':
    #x_d5, y_d5 = dataset5()
    #x_d5, y_d5 = preprocess(x_d5, y_d5)
    np.random.seed(42)
    loss_hist_batch = batch_steepest_gradient(epoch=200,lr=0.01)
    loss_hist_newton = newton_method(epoch=200,lr=0.01)
    #show_iter = 200
    #min_loss = min(np.min(loss_hist_batch), np.min(loss_hist_newton)) # ここ要修正かも
    plt.plot(np.abs(loss_hist_batch[:]-np.min(loss_hist_batch)), label='steepest')
    plt.plot(np.abs(loss_hist_newton[:]-np.min(loss_hist_newton)), label='newton')
    plt.legend()
    plt.yscale('log')
    plt.show()