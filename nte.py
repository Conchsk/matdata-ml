import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

raw_data = np.loadtxt('data-NTE-plus.csv', delimiter=',', skiprows=1)

X = raw_data[:, :10]
std = StandardScaler()
X = std.fit_transform(X)

# gradient
y0 = raw_data[:, 13]

best_C0 = 0
best_mse0 = 10000000
for it_C in range(1, 20000, 10):
    loo = LeaveOneOut()
    it_mse = 0.0
    for train, test in loo.split(X):
        it_mse += (SVR(C=it_C / 100, gamma='auto').fit(X=X[train], y=y0[train].reshape(-1)
                                                       ).predict(X[test]) - y0[test][0])**2
    if best_mse0 > it_mse:
        best_mse0 = it_mse
        best_C0 = it_C
print(best_C0)
model0 = SVR(C=best_C0 / 100, gamma='auto')
model0.fit(X, y0)
print(model0.score(X, y0))

# range
# y1 = raw_data[:, 12]
# y2 = raw_data[:, 12]
# y3 = raw_data[:, 13]

# best_C1 = 0
# best_mse1 = 10000000
# for it_C in range(1, 20000, 10):
#     loo = LeaveOneOut()
#     it_mse = 0.0
#     for train, test in loo.split(X):
#         it_mse += (SVR(C=it_C / 100, gamma='auto').fit(X=X[train], y=y1[train].reshape(-1)
#                                                        ).predict(X[test]) - y1[test][0])**2
#     if best_mse1 > it_mse:
#         best_mse1 = it_mse
#         best_C1 = it_C
# print(best_C1)
# model1 = SVR(C=best_C1 / 100, gamma='auto')
# model1.fit(X, y1)
# print(model1.score(X, y1))

# best_C2 = 0
# best_mse2 = 10000000
# for it_C in range(1, 20000, 10):
#     loo = LeaveOneOut()
#     it_mse = 0.0
#     for train, test in loo.split(X):
#         it_mse += (SVR(C=it_C / 100, gamma='auto').fit(X=X[train], y=y2[train].reshape(-1)
#                                                        ).predict(X[test]) - y2[test][0])**2
#     if best_mse2 > it_mse:
#         best_mse2 = it_mse
#         best_C2 = it_C
# print(best_C2)
# model2 = SVR(C=best_C2 / 100, gamma='auto')
# model2.fit(X, y2)
# print(model2.score(X, y2))

# y_hat = []
# for i in range(y3.shape[0]):
#     y_hat.append(model2.predict(X[i].reshape(1, -1))
#                  [0] - model1.predict(X[i].reshape(1, -1))[0])
# print(1 - np.linalg.norm(y3 - y_hat)**2 / np.linalg.norm(y3 - y3.mean())**2)

# plt.plot([25, 150], [25, 150], color='r')
# plt.scatter(y3, y_hat)
# plt.show()
