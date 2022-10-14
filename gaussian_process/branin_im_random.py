import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as WK,\
ExpSineSquared as ESS, RationalQuadratic as RQ, Matern as M
import warnings
warnings.filterwarnings("ignore")

# %matplotlib inline
def branin(x): 
    # x should be (2, )
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a*(x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s

def main():
    # generate data
    # uniform
    np.random.seed(0)
    data = []
    noise = []
    for i in range(20):
        xa = np.random.uniform(-5, -2, (1000, 1))
        xb = np.random.uniform(-2, 10, (1000, 1))
        c = np.random.choice(2, (1000,1), p =[0.2, 0.8])
        x1 = xa*c + xb * (1-c)
        x2 = np.random.uniform(0, 15, (1000, 1))
        noise.append(np.random.normal(0, 1, (1000,1)))             #(1000, 1)
        data.append(np.hstack((x1, x2)))   

    sigma = 5
    B_list = [1, 10, 100, 1000]
    T = 300
    Trs = 10
    d = 2
    for B in B_list:
        pulls_list = []
        L_trials = []
        error_list = []
        print("B = ", B)
        for trial in range(Trs):
            kernel = 1*RBF([1e-2, 1e2], (1e-5, 1e5))
            gp = GPR(kernel = kernel, optimizer = None)
            X, Y = [], []
            loss_list = []
            error_s = []
            L = 0
            pulls = np.zeros(T)
            for t in tqdm(range(T)):
                x = data[trial][t]                # (2, )
                f = branin(x)
                y = f + noise[trial][t,0] * sigma
                # f_mean, f_std = gp.predict(np.array([x]), return_std = True)
                # # if f_std > np.sqrt(2*sigma**2)*(B**(1./(3. + 2.)))*((t+1)**(-1./(3. + 2.))) or t == 0:
                # if f_std > theta or t ==0:
                if np.random.rand() < 0.5 or t < 5*d:
                    # theta = theta * (1+s)
                    L += B
                    pulls[t] = 1
                    X.append(x)
                    Y.append(y)
                    kernel = 1*RBF([1e-2, 1e2], (1e-5, 1e5))
                    gp = GPR(kernel = kernel, alpha = sigma**2, n_restarts_optimizer= 10)
                    gp.fit(np.array(X), np.array(Y).reshape(-1,1))
                # theta = theta * (1-s)
                f_mean, f_std = gp.predict(np.array([x]), return_std = True)
                p= f_mean.item()
                error = abs(p-f)
                L += error
                error_s.append(error)
                loss_list.append(L/(t+1))
            # if(pulls.sum() > 5):
            # print("trial %d added!"%trial)
            error_list.append(error_s)
            L_trials.append(loss_list)
            pulls_list.append(pulls)
            print("trial %d average pulls: %d" %(trial, pulls.sum()))
        np.save("./branin_imbalanced_loss_B_%d_T_%d_random"%(B, T), L_trials)
        np.save("./branin_imbalanced_error_B_%d_T_%d_random"%(B, T), error_list)

if __name__ == '__main__':
    main()