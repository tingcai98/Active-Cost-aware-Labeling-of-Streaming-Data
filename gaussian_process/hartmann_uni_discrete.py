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
    
def check_arm(x): 
  # x should be (6, )
    x = np.array(x < 0.5, dtype = int)
    return x[0]*(2**5) + x[1]*(2**4) + x[2]*(2**3) + x[3]*(2**2) + x[4]*(2**1) + x[5]

def main():
    # generate data
    # uniform
    np.random.seed(0)
    data = []
    noise = []
    for i in range(20):
        data.append(np.random.uniform(0, 1, (1000, 6)))
        noise.append(np.random.normal(0, 1, (1000,1)))
    
    sigma = 0.5
    B_list = [1, 10, 100, 1000]
    C_list = [0.5, 1, 2, 3, 4, 5]
    T = 300
    Trs = 10
    d = 6
    delta = 1
    for B in B_list:
        # if B == 1:
        #     C_l = C_list[1:6]
        # else:
        #     C_l = C_list
        for C in C_list:
            pulls_list = []
            L_trials = []
            error_list = []
            print("B = ", B, "C = ", C)
            for trial in range(Trs):
                kernel = 1*RBF([1e-2, 1e2, 1e-2, 1e2, 1e-2, 1e2], (1e-5, 1e5))
                gp = GPR(kernel = kernel, optimizer = None)
                X, Y = [], []
                loss_list = []
                error_s = []
                L = 0
                pulls = np.zeros(T)
                Mtk = np.zeros(2**d)
                Ntk = np.zeros(2**d)
                for t in tqdm(range(T)):
                    x = data[trial][t]                # (2, )
                    f = hartmann6_4(x.tolist())
                    y = f + noise[trial][t,0] * sigma
                    arm = check_arm(x)
                    Mtk[arm] += 1
                    # f_mean, f_std = gp.predict(np.array([x]), return_std = True)
                    # # if f_std > np.sqrt(2*sigma**2)*(B**(1./(3. + 2.)))*((t+1)**(-1./(3. + 2.))) or t == 0:
                    # if f_std > theta or t ==0:
                    # if np.random.rand() < 0.5 or t < 5*d:
                    if t < 5*d or Ntk[arm] == 0 or np.sqrt(2*np.log(2*(Mtk[arm]**3)/delta)/max(Ntk[arm], 1)) >  C * (B**(1./(3. +d))) * (t**(-1./(3.+d))):
                        # theta = theta * (1+s)
                        Ntk[arm] += 1
                        L += B
                        pulls[t] = 1
                        X.append(x)
                        Y.append(y)
                        kernel = 1*RBF([1e-2, 1e2, 1e-2, 1e2, 1e-2, 1e2], (1e-5, 1e5))
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
            np.save("./hartmann_uniform_loss_B_%d_T_%d_C_%d_discrete"%(B, T, C*100), L_trials)
            np.save("./hartmann_uniform_error_B_%d_T_%d_C_%d_discrete"%(B, T, C*100), error_list)

if __name__ == '__main__':
    main()