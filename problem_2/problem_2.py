import numpy as np
import matplotlib.pyplot as plt


def soft_threshold(q, mu):
    """
    Soft thresholding function
    argmin_w (1/2) ||w-mu||_2^2 + q ||w||_1
    """
    return np.sign(mu) * np.maximum(np.abs(mu) - q, 0)

def delta_psi(A,w,mu):
    """
    (A^T + A)(w - mu)
    """
    return (A.T + A) @ (w - mu)

def lasso(lambda_):
    """
    Lasso regression
    argmin_w (w-mu)^t A (w-mu) + lambda ||w||_1
    """
    A = np.array([[3,0.5],[0.5,1]])
    mu = np.array([[1, 2]]).T
    w = np.random.randn(2,1)
    gamma, _ = np.linalg.eig(A*2)
    gamma = np.max(gamma)
    print(gamma)
    ws = []
    ws.append(w)
    for _ in range(100):
        w = soft_threshold(lambda_ / gamma, w - delta_psi(A,w,mu)/gamma)
        ws.append(w)
    return ws

if __name__=='__main__':
    ws_2 = lasso(2)
    ws_4 = lasso(4)
    ws_6 = lasso(6)
    plt.plot([np.linalg.norm(b) for b in ws_2-ws_2[-1]], label=f'lambda={2}')
    plt.plot([np.linalg.norm(b) for b in ws_4-ws_4[-1]], label=f'lambda={4}')
    plt.plot([np.linalg.norm(b) for b in ws_6-ws_6[-1]], label=f'lambda={6}')
    plt.legend()
    plt.yscale('log')
    plt.savefig('problem_2.png')

