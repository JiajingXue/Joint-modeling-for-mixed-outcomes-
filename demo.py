## example
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

rho = 0.25  ### the AR correlation of X or error
n_train = 400
n = int((n_train / 3) * 5)  ### the total sample of size n, if you want 500 for training, then n should be 500 * (5/3)
p = 100  ### the dimension of X
p_0 = 30  ### the number of important features (X)
q = 3  ### the dimension of Z and S

seed = 0

hidden1 = 128
hidden2 = 128
hidden3 = 128
output = q
sigma = 2.50
lam = 0.065

mse = nn.MSELoss()
pos_true = coef(p, q, p_0, seed)[1]
x = getXande(rho, n, p, seed)
y = (getZandS(p, q, p_0, rho, n, 0.1, seed)[0])[:, 0].reshape(n, 1)
z = (getZandS(p, q, p_0, rho, n, 0.1, seed)[0])[:, 1].reshape(n, 1)
s = (getZandS(p, q, p_0, rho, n, 0.1, seed)[0])[:, 2].reshape(n, 1)  ## se for survival event
se = (getZandS(p, q, p_0, rho, n, 0.1, seed)[1]).reshape(n, 1)

x_train, x_valid, x_test = split(x)[0], split(x)[1], split(x)[2]
s_train, s_valid, s_test = split(s)[0], split(s)[1], split(s)[2]
se_train, se_valid, se_test = split(se)[0], split(se)[1], split(se)[2]  ## se for survival event
y_train, y_valid, y_test = split(y)[0], split(y)[1], split(y)[2]
z_train, z_valid, z_test = split(z)[0], split(z)[1], split(z)[2]

t1 = time.time()
n_1 = z_train.shape[0]
n_2 = z_valid.shape[0]
n_3 = z_test.shape[0]

Ss_train = tc.combinations(s_train.reshape(n_1))
Ss_valid = tc.combinations(s_valid.reshape(n_2))
Ss_test = tc.combinations(s_test.reshape(n_3))

Sse_train = tc.combinations(se_train.reshape(n_1))
Sse_valid = tc.combinations(se_valid.reshape(n_2))
Sse_test = tc.combinations(se_test.reshape(n_3))

Sy_train = tc.combinations(y_train.reshape(n_1))
Sy_valid = tc.combinations(y_valid.reshape(n_2))
Sy_test = tc.combinations(y_test.reshape(n_3))

Sz_train = tc.combinations(z_train.reshape(n_1))
Sz_valid = tc.combinations(z_valid.reshape(n_2))
Sz_test = tc.combinations(z_test.reshape(n_3))

# train
initnet = LTR(p, hidden1, hidden2, hidden3, output, sigma)
M_RankNet = M_LTRtrain(x_train, s_train, Ss_train, Sse_train, y_train, Sy_train, z_train, Sz_train,
                       x_valid, s_valid, Ss_valid, Sse_valid, y_valid, Sy_valid, z_valid, Sz_valid,
                       0.0, 100, lam, paint = True)
trained_net = M_RankNet[0]
t2 = time.time()
print("cost:", t2 - t1)

# evaluation
a = tc.zeros_like(M_RankNet[0].sparse.weight.data)
b = tc.zeros_like(M_RankNet[0].sparse.weight.data)
poss = tc.argwhere(M_RankNet[0].sparse.weight.data != 0)
a[poss] = 1
b[pos_true] = 1
cnf_matrix = confusion_matrix(b.tolist(), a.tolist())  ###(true, pred)
TPR = cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0])
FPR = cnf_matrix[0, 1] / (cnf_matrix[0, 0] + cnf_matrix[0, 1])
TP = cnf_matrix[1, 1]
FP = cnf_matrix[0, 1]
print(TPR, FPR)
print(TP, FP)

### refit for these responses
t1 = time.time()
selected_pos = poss.reshape(TP + FP).tolist()
selected_x_train = x_train[:, selected_pos]
selected_x_valid = x_valid[:, selected_pos]
selected_x_test = x_test[:, selected_pos]

re_hidden1 = 64
re_hidden2 = 64
re_hidden3 = 64

initnet_b = trained_B(TP + FP, re_hidden1, re_hidden2, re_hidden3, 1)
Sc_Net_y = Sb_train(selected_x_train, y_train, selected_x_valid, y_valid, 1e-3, 100, 0.01, paint = True)

initnet_c = trained_C(TP + FP, re_hidden1, re_hidden2, re_hidden3, 1)
Sc_Net_z = Sc_train(selected_x_train, z_train, se_train, selected_x_valid, z_valid, se_valid, 1e-2, 150, 0.02,
                    'continuous', paint = True)

initnet_c = trained_C(TP + FP, re_hidden1, re_hidden2, re_hidden3, 1)
Sb_Net_s = Sc_train(selected_x_train, s_train, se_train, selected_x_valid, s_valid, se_valid, 1e-2, 400, 0.01,
                    'survival', paint = True)
t2 = time.time()
print('time cost:', t2 - t1)

### evaluation
s_pred = Sb_Net_s[0](selected_x_test)
y_pred = Sc_Net_y[0](selected_x_test).data
z_pred = Sc_Net_z[0](selected_x_test)

print('C index:', concordance_index(s_pred, s_test, se_test))
print('AUC:', roc_auc_score(y_test, y_pred))
print('Rmse:', (mse(z_pred.reshape(n_3), z_test.reshape(n_3))) ** 0.5)
