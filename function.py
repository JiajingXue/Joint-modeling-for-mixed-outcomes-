from torch import nn
from itertools import combinations
import time
import numpy as np
import torch as tc
import random
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib as mpl

import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

from sortedcontainers import SortedList


#### Data generating process ####

def getXande(rho, n, p, seed):
    np.random.seed(seed)
    cov = np.zeros(shape=(p, p))
    mean = np.zeros(p)

    for i in range(p):
        for j in range(p):
            cov[i, j] = rho ** (abs(i - j))
    return np.random.multivariate_normal(mean=mean, cov=cov, size=n)  ## The shape of results is (n,p)

def coef(p, q, p_0, seed):
    random.seed(seed)
    np.random.seed(seed)
    position = random.sample(range(0, p), p_0)
    Beta = np.zeros(shape=(p, q))
    for k in position:
        for j in range(0, q):
            Beta[k, j] = random.uniform(-3, 3)
    if q > 1:
        Beta[:, 1] = 0.2 * Beta[:, 1]
    return Beta, position

# response
def getZandS(p,q,p_0,rho,n,censor_prop,seed):
    sig = nn.Sigmoid()
    random.seed(seed)
    np.random.seed(seed)
    # error terms
    error = getXande(0.75, n, q, seed) ## the dimension of error should be "q", which is the same as Z
    error[:,0] = np.zeros(n) ## binary response does not need error term
    # coefficients
    beta = coef(p,q,p_0,seed)[0]

    # covariates
    x = getXande(rho, n, p, seed)

    # response:
    ## continuous (Y[:,1:(q-2)]):
    y = (np.exp(np.sin(x))) @ beta + x @ (beta) + error
    ## survival (Y[:,q-1]):
    s_event = np.ones_like(y[:,q-1])
    censor_pos = np.random.choice(n, int(n*censor_prop),replace=False)
    s_event[censor_pos] = 0.
    (y[:,q-1])[censor_pos] = (y[:,q-1])[censor_pos] - np.random.uniform(0,10)
    ## binary (Y[:,0]):
    mu = np.array(sig(tc.Tensor(y[:,0])).data)
    y[:,0] = np.random.binomial(1,mu)
    return y, s_event

# train: valid: test = 3:1:1
def split(X):
    pos = list(range(n))
    np.random.seed(1)
    trainpos = np.random.choice(pos, int(n/5)*3,replace=False) ## train
    X_train = X[trainpos,:]
    pre_validpos = [i for i in pos if i not in trainpos]
    validpos = np.random.choice(pre_validpos, int(n/5), replace=False)
    X_valid = X[validpos,:]
    pre_testpos = [i for i in pre_validpos if i not in validpos]
    testpos = np.random.choice(pre_testpos, int(n/5), replace=False)
    X_test = X[testpos,:]
    return tc.Tensor(X_train),tc.Tensor(X_valid),tc.Tensor(X_test)


# Network parameters
# sparse layer
def sparse(sss, p):  ####### We initialize sparse layer from normal
    np.random.seed(sss)
    sp = np.random.normal(0, 0.02, size=p)
    sp = tc.Tensor(sp)
    return sp  # sp for sparse

class weight_sparse(nn.Module):  ####### in_features = p, the dimension of X
    def __init__(self, in_features):
        super(weight_sparse, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(sparse(6, self.in_features), requires_grad=True)

    def forward(self, input):
        x = input * self.weight
        return x

# weights
def weight(www, num_inputs, num_hiddens1, num_hiddens2, num_hiddens3, num_outputs): 
    np.random.seed(www)
    w1r = num_inputs
    w1c = num_hiddens1
    w1 = np.random.uniform(-0.1, 0.1, size=(w1c, w1r))
    w1 = tc.Tensor(w1)

    w2r = num_hiddens1
    w2c = num_hiddens2
    w2 = np.random.uniform(-0.1, 0.1, size=(w2c, w2r))
    w2 = tc.Tensor(w2)

    w3r = num_hiddens2
    w3c = num_hiddens3
    w3 = np.random.uniform(-0.1, 0.1, size=(w3c, w3r))
    w3 = tc.Tensor(w3)

    w4r = num_hiddens3
    w4c = num_outputs
    w4 = np.random.uniform(-0.1, 0.1, size=(w4c, w4r))
    w4 = tc.Tensor(w4)
    return w1, w2, w3, w4

# biases
def bias(iii, num_hiddens1, num_hiddens2, num_hiddens3, num_outputs):
    np.random.seed(iii)
    b1 = np.random.uniform(-0.1, 0.1, size=num_hiddens1)
    b2 = np.random.uniform(-0.1, 0.1, size=num_hiddens2)
    b3 = np.random.uniform(-0.1, 0.1, size=num_hiddens3)
    b4 = np.random.uniform(-0.1, 0.1, size=num_outputs)

    b1 = tc.Tensor(b1)
    b2 = tc.Tensor(b2)
    b3 = tc.Tensor(b3)
    b4 = tc.Tensor(b4)
    return b1, b2, b3, b4


# Network structure
class LTR(nn.Module):  ### predictor
    def __init__(self, p, hidden1, hidden2, hidden3, out, sigma):
        super(LTR, self).__init__()
        # activation
        self.prl = nn.PReLU()
        self.sig = nn.Sigmoid()

        # sparse connection
        self.sparse = weight_sparse(p)

        # tuning: sigma
        self.sigma = sigma

        self.sc1 = nn.Linear(p, hidden1)
        self.sc2 = nn.Linear(hidden1, hidden2)
        self.sc3 = nn.Linear(hidden2, hidden3)
        self.sc4 = nn.Linear(hidden3, out)

        self.sc1.weight.data = weight(6, p, hidden1, hidden2, hidden3, out)[0]
        self.sc2.weight.data = weight(6, p, hidden1, hidden2, hidden3, out)[1]
        self.sc3.weight.data = weight(6, p, hidden1, hidden2, hidden3, out)[2]
        self.sc4.weight.data = weight(6, p, hidden1, hidden2, hidden3, out)[3]

        self.sc1.bias.data = bias(6, hidden1, hidden2, hidden3, out)[0]
        self.sc2.bias.data = bias(6, hidden1, hidden2, hidden3, out)[1]
        self.sc3.bias.data = bias(6, hidden1, hidden2, hidden3, out)[2]
        self.sc4.bias.data = bias(6, hidden1, hidden2, hidden3, out)[3]

    def forward(self, x):
        n_0 = x.shape[0]
        x = self.sparse(x)
        x = self.prl(self.sc1(x))
        x = self.prl(self.sc2(x))
        x = self.prl(self.sc3(x))
        x = self.sig(self.sc4(x))
        phts = tc.combinations(x[:, 0].reshape(n_0), 2)
        phtz1 = tc.combinations(x[:, 1].reshape(n_0), 2)
        phtz2 = tc.combinations(x[:, 2].reshape(n_0), 2)

        return self.sig(self.sigma * (phts[:, 0] - phts[:, 1])), self.sig(
            self.sigma * (phtz1[:, 0] - phtz1[:, 1])), self.sig(self.sigma * (phtz2[:, 0] - phtz2[:, 1])), x  #

# Refit network
# Refit for continuous and survival responses
class trained_C(nn.Module):
    def __init__(self, selected_p, re_hidden1, re_hidden2, re_hidden3, re_out):
        super(trained_C, self).__init__()
        self.prl = nn.PReLU()
        self.sig = nn.Sigmoid()

        self.sc1 = nn.Linear(selected_p, re_hidden1)
        self.sc2 = nn.Linear(re_hidden1, re_hidden2)
        self.sc3 = nn.Linear(re_hidden2, re_hidden3)
        self.sc4 = nn.Linear(re_hidden3, re_out)

        self.sc1.weight.data = weight(7, selected_p, re_hidden1, re_hidden2, re_hidden3, re_out)[0]
        self.sc2.weight.data = weight(7, selected_p, re_hidden1, re_hidden2, re_hidden3, re_out)[1]
        self.sc3.weight.data = weight(7, selected_p, re_hidden1, re_hidden2, re_hidden3, re_out)[2]
        self.sc4.weight.data = weight(7, selected_p, re_hidden1, re_hidden2, re_hidden3, re_out)[3]

        self.sc1.bias.data = bias(7, re_hidden1, re_hidden2, re_hidden3, re_out)[0]
        self.sc2.bias.data = bias(7, re_hidden1, re_hidden2, re_hidden3, re_out)[1]
        self.sc3.bias.data = bias(7, re_hidden1, re_hidden2, re_hidden3, re_out)[2]
        self.sc4.bias.data = bias(7, re_hidden1, re_hidden2, re_hidden3, re_out)[3]

    def forward(self, x):
        n_0 = x.shape[0]
        x = self.prl(self.sc1(x))
        x = self.prl(self.sc2(x))
        x = self.prl(self.sc3(x))
        x = self.prl(self.sc4(x))

        return x

# Refit for binary response
class trained_B(nn.Module):
    def __init__(self, selected_p, re_hidden1, re_hidden2, re_hidden3, re_out):
        super(trained_B, self).__init__()
        self.prl = nn.PReLU()
        self.sig = nn.Sigmoid()

        self.sc1 = nn.Linear(selected_p, re_hidden1)
        self.sc2 = nn.Linear(re_hidden1, re_hidden2)
        self.sc3 = nn.Linear(re_hidden2, re_hidden3)
        self.sc4 = nn.Linear(re_hidden3, re_out)

        self.sc1.weight.data = weight(7, selected_p, re_hidden1, re_hidden2, re_hidden3, re_out)[0]
        self.sc2.weight.data = weight(7, selected_p, re_hidden1, re_hidden2, re_hidden3, re_out)[1]
        self.sc3.weight.data = weight(7, selected_p, re_hidden1, re_hidden2, re_hidden3, re_out)[2]
        self.sc4.weight.data = weight(7, selected_p, re_hidden1, re_hidden2, re_hidden3, re_out)[3]

        self.sc1.bias.data = bias(7, re_hidden1, re_hidden2, re_hidden3, re_out)[0]
        self.sc2.bias.data = bias(7, re_hidden1, re_hidden2, re_hidden3, re_out)[1]
        self.sc3.bias.data = bias(7, re_hidden1, re_hidden2, re_hidden3, re_out)[2]
        self.sc4.bias.data = bias(7, re_hidden1, re_hidden2, re_hidden3, re_out)[3]

    def forward(self, x):
        n_0 = x.shape[0]
        x = self.prl(self.sc1(x))
        x = self.prl(self.sc2(x))
        x = self.prl(self.sc3(x))
        x = self.prl(self.sc4(x))

        return self.sig(x)

# Loss function
# Loss function for continuous and binary responses
def objt(Pij_hat, S):
    n_00 = S.shape[0]
    score = tc.zeros(n_00)
    yi = S[:, 0]
    yj = S[:, 1]
    score[yi > yj] = 1
    score[yi <= yj] = -1

    Pij = 0.5 * (1 + score)
    func = nn.BCELoss()
    loss = func(Pij_hat, Pij)
    return loss

# Loss function for survival response
def objt_s(Pij_hat, S, SE):
    n_00 = S.shape[0]
    score = tc.zeros(n_00)
    yi = S[:, 0]
    yj = S[:, 1]
    score[yi > yj] = 1
    score[yi <= yj] = -1

    Pij = 0.5 * (1 + score)
    Pij[SE[:, 1] == 0] = 0
    func = nn.BCELoss()
    loss = func(Pij_hat, Pij)
    return loss

# jumps (weights) in the Kaplan-Meier estimator (for refitting)
def Weight(time, censor_indicator):
    ### time: observed time of shape (n,1)
    ### censor_indicator of shape (n,1)

    n = time.shape[0]
    order = tc.sort(time.reshape(n))[1].tolist()  ### the order of y from small to large

    delta = tc.zeros(n)  ### ordered censor indicator with the order of y
    k = 0
    for j in order:
        delta[k] = censor_indicator[j]
        k = k + 1

    weight = tc.zeros(n)
    weight[0] = delta[0] / n
    for i in range(1, n):
        tmp = tc.ones(i)
        for l in range(i):
            tmp[l] = ((n - l - 1) / (n - l)) ** delta[l].item()
        weight[i] = (delta[i] / (n - i)) * tc.prod(tmp)

    WEIGHT = tc.zeros(n) 

    q = 0
    for p in order:
        WEIGHT[p] = weight[q]
        q = q + 1

    return WEIGHT

# proximal gradient with MCP
def soft_thresholding(X, lam):
    return X.sign() * (X.abs() > lam) * (X.abs() - lam)

def prox_mcp(X, lam, gamma=3, penalty="mcp"):
    if penalty == "mcp":
        return (X.abs() < gamma * lam) * soft_thresholding(X, lam) / (1 - 1 / gamma) + \
               (X.abs() >= gamma * lam) * X
    else:
        return soft_thresholding(X, lam)


def M_LTRtrain(train_x, train_y, train_Sy, train_SEy, train_z1, train_Sz1, train_z2, train_Sz2,
               eval_x, eval_y, eval_Sy, eval_SEy, eval_z1, eval_Sz1, eval_z2, eval_Sz2, \
               L2, Num_Epochs, lam, paint): 
    n_4 = train_y.shape[0]
    n_5 = eval_y.shape[0]
    loss_train_y_list = []
    loss_test_y_list = []
    Kendall_tau_train_y = []
    Kendall_tau_eval_y = []

    loss_train_z1_list = []
    loss_test_z1_list = []
    Kendall_tau_train_z1 = []
    Kendall_tau_eval_z1 = []

    loss_train_z2_list = []
    loss_test_z2_list = []
    Kendall_tau_train_z2 = []
    Kendall_tau_eval_z2 = []

    net = initnet
    opt = optim.Adam([
        {'params': net.sparse.parameters(), 'weight_decay': 0, 'lr': 0.02, 'betas': (0.9, 0.999), 'eps': 1e-08,
         'amsgrad': False},
        {'params': net.sc1.parameters(), 'weight_decay': L2, 'lr': 0.01, 'betas': (0.9, 0.999), 'eps': 1e-08,
         'amsgrad': False},
        {'params': net.sc2.parameters(), 'weight_decay': L2, 'lr': 0.01, 'betas': (0.9, 0.999), 'eps': 1e-08,
         'amsgrad': False},
        {'params': net.sc3.parameters(), 'weight_decay': L2, 'lr': 0.01, 'betas': (0.9, 0.999), 'eps': 1e-08,
         'amsgrad': False},
    ])

    for epoch in range(Num_Epochs + 1):
        net.train()
        pij_pred_y = net(train_x)[0]
        output_train_y = (net(train_x)[3])[:, 0]
        opt.zero_grad()
        lossy = objt_s(pij_pred_y, train_Sy, train_SEy)
        # lossy = objt(pij_pred_y, train_Sy)
        loss_train_y_list.append(lossy)
        Kendall_tau_train_y.append(
            stats.kendalltau((output_train_y.reshape(n_4)).tolist(), (train_y.reshape(n_4)).tolist())[0])

        pij_pred_z1 = net(train_x)[1]
        output_train_z1 = (net(train_x)[3])[:, 1]
        opt.zero_grad()
        lossz1 = objt(pij_pred_z1, train_Sz1)
        loss_train_z1_list.append(lossz1)
        Kendall_tau_train_z1.append(
            stats.kendalltau((output_train_z1.reshape(n_4)).tolist(), (train_z1.reshape(n_4)).tolist())[0])

        pij_pred_z2 = net(train_x)[2]
        output_train_z2 = (net(train_x)[3])[:, 2]
        opt.zero_grad()
        lossz2 = objt(pij_pred_z2, train_Sz2)
        loss_train_z2_list.append(lossz2)
        tau_train_z2 = stats.kendalltau((output_train_z2.reshape(n_4)).tolist(), (train_z2.reshape(n_4)).tolist())[0]
        Kendall_tau_train_z2.append(tau_train_z2)
        loss = (lossy + lossz1 + lossz2) / 3

        ### test evaluation
        pij_predt_y = net(eval_x)[0]
        output_eval_y = (net(eval_x)[3])[:, 0]
        lossty = objt_s(pij_predt_y, eval_Sy, eval_SEy)
        # lossty= objt(pij_predt_y, eval_Sy)
        loss_test_y_list.append(lossty)
        Kendall_tau_eval_y.append(
            stats.kendalltau((output_eval_y.reshape(n_5)).tolist(), (eval_y.reshape(n_5)).tolist())[0])

        pij_predt_z1 = net(eval_x)[1]
        output_eval_z1 = (net(eval_x)[3])[:, 1]
        losstz1 = objt(pij_predt_z1, eval_Sz1)
        loss_test_z1_list.append(losstz1)
        tau_eval_z1 = stats.kendalltau((output_eval_z1.reshape(n_5)).tolist(), (eval_z1.reshape(n_5)).tolist())[0]
        Kendall_tau_eval_z1.append(tau_eval_z1)

        pij_predt_z2 = net(eval_x)[2]
        output_eval_z2 = (net(eval_x)[3])[:, 2]
        losstz2 = objt(pij_predt_z2, eval_Sz2)
        loss_test_z2_list.append(losstz2)
        tau_eval_z2 = stats.kendalltau((output_eval_z2.reshape(n_5)).tolist(), (eval_z2.reshape(n_5)).tolist())[0]
        Kendall_tau_eval_z2.append(tau_eval_z2)
        losst = (lossty + losstz1 + losstz2) / 3

        loss.backward()
        opt.step()
        if epoch > 20:
            net.sparse.weight.data = prox_mcp(net.sparse.weight.data, lam, gamma=3, penalty="mcp")

    if paint == True:
        loss_train_y_list = tc.Tensor(loss_train_y_list)
        loss_test_y_list = tc.Tensor(loss_test_y_list)
        Kendall_tau_train_y = tc.Tensor(Kendall_tau_train_y)
        Kendall_tau_eval_y = tc.Tensor(Kendall_tau_eval_y)
        loss_train_z1_list = tc.Tensor(loss_train_z1_list)
        loss_test_z1_list = tc.Tensor(loss_test_z1_list)
        Kendall_tau_train_z1 = tc.Tensor(Kendall_tau_train_z1)
        Kendall_tau_eval_z1 = tc.Tensor(Kendall_tau_eval_z1)
        loss_train_z2_list = tc.Tensor(loss_train_z2_list)
        loss_test_z2_list = tc.Tensor(loss_test_z2_list)
        Kendall_tau_train_z2 = tc.Tensor(Kendall_tau_train_z2)
        Kendall_tau_eval_z2 = tc.Tensor(Kendall_tau_eval_z2)

        plt.subplot(231)
        plt.plot(loss_train_y_list, label='train',color = 'b',linestyle = 'dashdot')
        plt.plot(loss_test_y_list, label='eval',color = 'c',linestyle = 'dotted')
        plt.title('Loss for S')
        plt.subplot(234)
        plt.plot(Kendall_tau_train_y, label='train',color = 'sienna',linestyle = '-')
        plt.plot(Kendall_tau_eval_y, label='eval',color = 'tomato',linestyle = '-')
        plt.title('Kendall_tau for S')
        plt.ylim(0,1)
        plt.subplot(232)
        plt.plot(loss_train_z1_list, label='train',color = 'b',linestyle = 'dashdot')
        plt.plot(loss_test_z1_list, label='eval',color = 'c',linestyle = 'dotted')
        plt.title('Loss for Z1')
        plt.subplot(235)
        plt.plot(Kendall_tau_train_z1, label='train',color = 'sienna',linestyle = '-')
        plt.plot(Kendall_tau_eval_z1, label='eval',color = 'tomato',linestyle = '-')
        plt.title('Kendall_tau for Z1')
        plt.ylim(0,1)
        plt.subplot(233)
        plt.plot(loss_train_z2_list, label='train',color = 'b',linestyle = 'dashdot')
        plt.plot(loss_test_z2_list, label='eval',color = 'c',linestyle = 'dotted')
        plt.title('Loss for Z2')
        plt.subplot(236)
        plt.plot(Kendall_tau_train_z2, label='train',color = 'sienna',linestyle = '-')
        plt.plot(Kendall_tau_eval_z2, label='eval',color = 'tomato',linestyle = '-')
        plt.title('Kendall_tau for Z2')
        plt.ylim(0,1)

        plt.legend(prop = {'size':8})
        plt.show()
    else:
        pass

    return (net, Kendall_tau_eval_y[-1], Kendall_tau_eval_z1[-1], Kendall_tau_eval_z2[-1], losst)

## if we want to refit using the selected variables:

# for continuous and survival:
def Sc_train(train_x, train_y, train_ye, eval_x, eval_y, eval_ye, L2, Num_Epochs, lr, dtype, paint):
    mse = nn.MSELoss()
    n_4 = train_y.shape[0]
    n_5 = eval_y.shape[0]
    loss_train_y_list = []
    loss_test_y_list = []

    net = initnet_c
    opt = optim.Adam([
        {'params': net.sc1.parameters(), 'weight_decay': L2, 'lr': lr, 'betas': (0.999, 0.999), 'eps': 1e-08,
         'amsgrad': False},
        {'params': net.sc2.parameters(), 'weight_decay': L2, 'lr': lr, 'betas': (0.999, 0.999), 'eps': 1e-08,
         'amsgrad': False},
        {'params': net.sc3.parameters(), 'weight_decay': L2, 'lr': lr, 'betas': (0.999, 0.999), 'eps': 1e-08,
         'amsgrad': False},
    ])
    if dtype == 'continuous':
        for epoch in range(Num_Epochs + 1):
            net.train()
            pred_y = net(train_x)
            opt.zero_grad()
            lossy = mse(pred_y.reshape(n_4), train_y.reshape(n_4))
            loss_train_y_list.append(lossy)

            predt_y = net(eval_x)
            lossty = mse(predt_y.reshape(n_5), eval_y.reshape(n_5))
            loss_test_y_list.append(lossty)

            lossy.backward()
            opt.step()
    else:
        weight_train = Weight(train_y, train_ye.reshape(n_4))
        weight_valid = Weight(eval_y, eval_ye.reshape(n_5))
        for epoch in range(Num_Epochs + 1):
            net.train()
            pred_y = net(train_x)
            opt.zero_grad()
            lossy = tc.Tensor(tc.dot(weight_train, tc.square(train_y.reshape(n_4) - pred_y.reshape(n_4))))
            loss_train_y_list.append(lossy)

            predt_y = net(eval_x)
            lossty = tc.Tensor(tc.dot(weight_valid, tc.square(eval_y.reshape(n_5) - predt_y.reshape(n_5))))
            loss_test_y_list.append(lossty)

            lossy.backward()
            opt.step()
    if paint == True:
        loss_train_y_list = tc.Tensor(loss_train_y_list)
        loss_test_y_list = tc.Tensor(loss_test_y_list)

        plt.plot(loss_train_y_list, label='train',color = 'b',linestyle = 'dashdot')
        plt.plot(loss_test_y_list, label='eval',color = 'c',linestyle = 'dotted')
        plt.title(dtype)

        plt.legend(prop = {'size':8})
        plt.show()

    return (net, lossty)

# for binary:
def Sb_train(train_x, train_y, eval_x, eval_y, L2, Num_Epochs, lr, paint):  ### Sb for single binary
    bfun = nn.BCELoss()
    n_4 = train_y.shape[0]
    n_5 = eval_y.shape[0]
    loss_train_y_list = []
    loss_test_y_list = []

    net = initnet_b
    opt = optim.Adam([
        {'params': net.sc1.parameters(), 'weight_decay': L2, 'lr': lr, 'betas': (0.995, 0.99), 'eps': 1e-08,
         'amsgrad': False},
        {'params': net.sc2.parameters(), 'weight_decay': L2, 'lr': lr, 'betas': (0.995, 0.99), 'eps': 1e-08,
         'amsgrad': False},
        {'params': net.sc3.parameters(), 'weight_decay': L2, 'lr': lr, 'betas': (0.995, 0.99), 'eps': 1e-08,
         'amsgrad': False},
    ])
    for epoch in range(Num_Epochs + 1):
        net.train()
        pred_y = net(train_x)
        opt.zero_grad()
        lossy = bfun(pred_y.reshape(n_4), train_y.reshape(n_4))
        loss_train_y_list.append(lossy)

        predt_y = net(eval_x)
        lossty = bfun(predt_y.reshape(n_5), eval_y.reshape(n_5))
        loss_test_y_list.append(lossty)

        lossy.backward()
        opt.step()
    if paint == True:
        loss_train_y_list = tc.Tensor(loss_train_y_list)
        loss_test_y_list = tc.Tensor(loss_test_y_list)

        plt.plot(loss_train_y_list, label='train',color = 'b',linestyle = 'dashdot')
        plt.plot(loss_test_y_list, label='eval',color = 'c',linestyle = 'dotted')
        plt.title('binary')

        plt.legend(prop = {'size':8})
        plt.show()

    return (net, lossty)
else:
    pass
