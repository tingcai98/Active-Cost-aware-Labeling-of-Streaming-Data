import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as WK,\
ExpSineSquared as ESS, RationalQuadratic as RQ, Matern as M
import warnings
warnings.filterwarnings("ignore")

def main():
    f = open("davisdata", "r")
    data = []
    for x in f:
        data.append(np.array(x.split()).astype(np.float))
    f.close()
    data = np.array(data)
    label = np.load("./davisdata_label.npy")

    np.random.seed(0)

    
    sigma = 10
    B_list = [1, 10, 100, 1000]
    T = data.shape[0]
    Trs = 5
    noise = []
    d = data.shape[1]

    for i in range(Trs):
        noise.append(np.random.normal(0,1,(T,1)))

    C_list = [0.1, 0.25, 0.5, 1, 1.5]

    for B in B_list:
        for C in C_list:
            pulls_list = []
            L_trials = []
            error_list = []
            print("B = ", B, " C = ", C)
            for trial in range(Trs):
                kernel = 1*RBF([1e-2, 1e2, 1e-2], (1e-5, 1e5))
                gp = GPR(kernel = kernel, optimizer = None)
                X, Y = [], []
                loss_list = []
                error_s = []
                L = 0
                pulls = np.zeros(T)
                for t in tqdm(range(T)):
                    x = data[t]                # (9, )
                    f = label[t]
                    y = f + noise[trial][t,0] * sigma
                    f_mean, f_std = gp.predict(np.array([x]), return_std = True)
                    if f_std > C *np.sqrt(2*sigma**2)*(B**(1./(3. + d)))*((t+1)**(-1./(3. + d))) or t < 5*d:
                    # if f_std > theta or t ==0:
                    # if np.random.rand() < 0.5 or t < 5*d:
                        # theta = theta * (1+s)
                        L += B
                        pulls[t] = 1
                        X.append(x)
                        Y.append(y)
                        kernel = 1*RBF([1e-2, 1e2, 1e-2], (1e-5, 1e5))
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
            np.save("./supernova_uniform_loss_B_%d_T_%d_C_%d_ours"%(B, T, C*100), L_trials)
            np.save("./supernova_uniform_error_B_%d_T_%d_C_%d_ours"%(B, T, C*100), error_list)

if __name__ == '__main__':
    main()