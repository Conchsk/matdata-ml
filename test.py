import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

from MyBayesianOptimization import MyBayesianOptimization
from MySVR import MySVR

import math

import matplotlib.pyplot as plt

# raw_data = np.loadtxt('data-NTE-min.csv', delimiter=',', skiprows=1)[:, :11]
# X = raw_data[:, :7]
# y = raw_data[:, 10]
# model = MySVR()
# model.fit(X, y)
# # print(model.score(X, y))


def visualization_2d(model):
    X = np.arange(-2, 10 + 1, 1)
    y_ei = model.acq(X.reshape(-1, 1))
    y_pi = model.acq_pi(X.reshape(-1, 1))
    y_ucb = model.acq_ucb(X.reshape(-1, 1))
    plt.figure(figsize=(6, 3))
    plt.plot(X, y_ei.reshape(-1), color='r', label="EI")
    plt.plot(X, y_pi.reshape(-1), color='g', label="PI")
    plt.plot(X, y_ucb.reshape(-1), color='b', label="UCB")
    plt.legend()
    plt.show()


def func(X):
    return np.exp(-(X - 2)**2) + np.exp(-(X - 6)**2/10) + 1 / (X**2 + 1)


model = MyBayesianOptimization(bounds=[(-2, 10)], ee_ration=0.1)
model.register([np.array([-2])], [func(np.array([-2]))])
model.register([np.array([10])], [func(np.array([10]))])
# model.register([np.array([0])], [func(np.array([0]))])
# visualization_2d(model)
model.visualization_2d(func, 0)
# for i in range(1,10,1):
#     next_X = model.suggest()
#     print(next_X)
#     model.register([next_X], [func(next_X)])
#     # model.visualization_2d(func, i)
#     visualization_2d(model)
