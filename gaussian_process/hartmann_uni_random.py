import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as WK,\
ExpSineSquared as ESS, RationalQuadratic as RQ, Matern as M
import warnings
warnings.filterwarnings("ignore")

# %matplotlib inline
def hartmann6_4(x):
    """ Hartmann function in 3D. """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    return -hartmann6_4_alpha(x, alpha)

def hartmann6_4_alpha(x, alpha):
    """ Hartmann function in 3D. """
#     pt = np.array([x[1][1]/10.0,
#                  (x[0][0] - 224)/100.0,
#                  x[3][0]/92.0,
#                  x[2][0],
#                  x[3][1]/92.0,
#                  x[1][0]/10.0,
#                 ])
    A = np.array([[  10,   3,   17, 3.5, 1.7,  8],
                [0.05,  10,   17, 0.1,   8, 14],
                [   3, 3.5,  1.7,  10,  17,  8],
                [  17,   8, 0.05,  10, 0.1, 14]], dtype=np.float64)
    P = 1e-4 * np.array([[1312, 1696, 5569,  124, 8283, 5886],
                       [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650],
                       [4047, 8828, 8732, 5743, 1091,  381]], dtype=np.float64)
    log_sum_terms = (A * (P - x)**2).sum(axis=1)
    return alpha.dot(np.exp(-log_sum_terms))


def main():
    # generate data uniform
    np.random.seed(0)
    data = []
    noise = []
    for i in range(20):
        data.append(np.random.uniform(0, 1, (1000, 6)))
        noise.append(np.random.normal(0, 1, (1000,1)))  
    sigma = 0.5
    B_list = [1, 10, 100, 1000]
    T = 300
    Trs = 10
    d = 6
    for B in B_list:
        pulls_list = []
        L_trials = []
        error_list = []
        print("B = ", B)
        for trial in range(Trs):
            kernel = 1*RBF([1e-2, 1e2, 1e-2, 1e2, 1e-2, 1e2], (1e-5, 1e5))
            gp = GPR(kernel = kernel, optimizer = None)
            X, Y = [], []
            loss_list = []
            error_s = []
            L = 0
            pulls = np.zeros(T)
            for t in tqdm(range(T)):
                x = data[trial][t]                # (2, )
                f = hartmann6_4(x.tolist())
                y = f + noise[trial][t,0] * sigma
                if np.random.rand() < 0.5 or t < 5*d:
                    L += B
                    pulls[t] = 1
                    X.append(x)
                    Y.append(y)
                    kernel = 1*RBF([1e-2, 1e2, 1e-2, 1e2, 1e-2, 1e2], (1e-5, 1e5))
                    gp = GPR(kernel = kernel, alpha = sigma**2, n_restarts_optimizer= 10)
                    gp.fit(np.array(X), np.array(Y).reshape(-1,1))
                f_mean, f_std = gp.predict(np.array([x]), return_std = True)
                p= f_mean.item()
                error = abs(p-f)
                L += error
                error_s.append(error)
                loss_list.append(L/(t+1))
            error_list.append(error_s)
            L_trials.append(loss_list)
            pulls_list.append(pulls)
            print("trial %d average pulls: %d"%(trial, pulls.sum()))
        np.save("./hartmann_uniform_loss_B_%d_T_%d_random"%(B, T), L_trials)
        np.save("./hartmann_uniform_error_B_%d_T_%d_random"%(B, T), error_list)

if __name__ == '__main__':
    main()