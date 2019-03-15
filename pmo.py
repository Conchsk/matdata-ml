import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

raw_data = np.loadtxt('data-PMO-GCr15.csv', delimiter=',', skiprows=1)

X = raw_data[:, :5]
std = StandardScaler()
X = std.fit_transform(X)

# gradient
y = raw_data[:, 5]

best_C = 0
best_mse = 10000000
for it_C in range(1, 20000, 10):
    loo = LeaveOneOut()
    it_mse = 0.0
    for train, test in loo.split(X):
        it_mse += (SVR(C=it_C / 100, gamma='auto').fit(X=X[train], y=y[train].reshape(-1)
                                                       ).predict(X[test]) - y[test][0])**2
    if best_mse > it_mse:
        best_mse = it_mse
        best_C = it_C
print(best_C, best_mse)
model = SVR(C=best_C / 100, gamma='auto')
model.fit(X, y)
print(model.score(X, y))
