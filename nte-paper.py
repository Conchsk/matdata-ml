import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from main import tunning
from MyGridSearch import MyGridSearch

import matplotlib.pyplot as plt

''' train '''
# read data
data = np.loadtxt('paper-train.csv', delimiter=',', skiprows=1)
X = data[:, 1: 11]
scaler_X = StandardScaler()
scaler_X.fit(X)
X_std = scaler_X.transform(X)
print(f'X_std:\n{X_std}')
print('\n--------------------\n')

y_start = data[:, 12]
scaler_y_start = StandardScaler()
scaler_y_start.fit(y_start.reshape(-1, 1))
# y_start_std = scaler_y_start.transform(y_start.reshape(-1, 1)).reshape(-1)
y_start_std = y_start
print(f'y_start_std:\n{y_start_std}')
print('\n--------------------\n')

y_end = data[:, 13]
scaler_y_end = StandardScaler()
scaler_y_end.fit(y_end.reshape(-1, 1))
# y_end_std = scaler_y_end.transform(y_end.reshape(-1, 1)).reshape(-1)
y_end_std = y_end
print(f'y_end_std:\n{y_end_std}')
print('\n--------------------\n')

# train start model
# opt_kernel, opt_C = tunning(X_std, y_start_std)
opt_kernel, opt_C = 'rbf', 100
model_start = SVR(kernel=opt_kernel, C=opt_C, gamma='auto')
model_start.fit(X_std, y_start_std)
print(f'model_start score:\n{model_start.score(X_std, y_start_std)}')
print('\n--------------------\n')

# train stop model
# opt_kernel, opt_C = tunning(X_std, y_end_std)
opt_kernel, opt_C = 'rbf', 100
model_end = SVR(kernel=opt_kernel, C=opt_C, gamma='auto')
model_end.fit(X_std, y_end_std)
print(f'model_end score:\n{model_end.score(X_std, y_end_std)}')
print('\n--------------------\n')


def model_range_predict(X):
    y_start = model_start.predict(scaler_X.transform(X))
    y_end = model_end.predict(scaler_X.transform(X))
    return y_end - y_start


y_real = y_end - y_start
y_hat = model_range_predict(X)
label_x = np.arange(20, 150, 1)
plt.plot(label_x, label_x, color='r')
plt.scatter(y_real, y_hat)
plt.show()
print(1-np.linalg.norm(y_hat-y_real)**2/np.linalg.norm(y_real-y_real.mean())**2)
print(model_start.predict(X_std)-y_start)
print(model_end.predict(X_std)-y_end)


# ''' grid search '''
# gs = MyGridSearch()
# opt_X, opt_y = gs.search(lambda x: model_range_predict([x]), [(0, 1) for i in range(X.shape[1])], [4 for i in range(X.shape[1])],
#                          [lambda x: abs(x[:8].sum() - 1) < 1e-6, lambda x: abs(x[8:].sum() - 1) < 1e-6])
# print(opt_X, opt_y)

# # ''' bayesian optimization '''
# # bys_opt = MyBayesianOptimization()

# # ''' predict '''
# # # read data
# # data = np.loadtxt('paper-predict.csv', delimiter=',', skiprows=1)
# # X = data[:, 1: 11]
# # y_range = data[:, 11]

# # # standardization
# # X_std = scaler.transform(X)

# # # predict
# # print(y_range)
# # print(model_end.predict(X_std) - model_start.predict(X_std))


# # y_range = filtered_data[:, 15]
# # bys_opt = MyBayesianOptimization(bounds=[(0, 1) for i in range(X.shape[1])],
# #                                  constraints=[{'type': 'eq', 'fun': lambda X: X[:8].sum() - 1},
# #                                               {'type': 'eq', 'fun': lambda X: X[8:].sum() - 1}])
# # bys_opt.register(X, y_range)


# # predict_data = np.loadtxt('paper-predict.csv', delimiter=',', skiprows=1)
# # X = predict_data[:, 1: 11]
# # std_X = scaler.transform(X)
# # y_range = predict_data[:, 11]
# # print(y_range)
# # print(model_start.predict(std_X))
# # print(model_end.predict(std_X))


# # for i in range(1000):
# #     opt_x = bys_opt.suggest()
# #     std_opt_x = scaler.transform([opt_x])
# #     y_hat = model_end.predict(std_opt_x)[0] - model_start.predict(std_opt_x)[0]
# #     bys_opt.register([opt_x], [y_hat])
# #     print(f'current registered point is X = \n{opt_x}\n y_hat = {y_hat}')
