import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVC


def tunning(X, y):
    opt_C = 0
    min_mse = 10000000
    for it_C in range(1, 20001, 10):
        model = SVR(C=it_C / 100, gamma='auto')
        it_mse = 0
        loo = LeaveOneOut()
        for train, test in loo.split(X):
            model.fit(X[train], y[train].reshape(-1))
            it_mse += (model.predict(X[test])[0] - y[test][0])**2
        if min_mse > it_mse:
            min_mse = it_mse
            opt_C = it_C / 100
    print(f'optimal C is {opt_C}',
          f'with R^2 = {1-min_mse/np.linalg.norm(y-y.mean())**2}')
    return opt_C


if __name__ == '__main__':
    raw_data = np.loadtxt('data-PMO-GCr15.csv', delimiter=',',
                          skiprows=1)[[0, 2, 3, 4, 5, 8, 9, 11, 12, 6, 7, 10]]
    X = raw_data[:, [0, 1, 2, 3]]
    std = StandardScaler()
    X = std.fit_transform(X)
    y = raw_data[:, 5]
    opt_C = tunning(X, y)
    # # opt_C = 3.01
    model = SVR(C=opt_C, gamma='auto')
    model.fit(X, y)
    print(model.score(X, y))


# 10 4061/0.43806226031254236
# 11 19951/0.557826453682837
# 12 19831/0.5425249017595752
# 13 4631/0.044084679151381034
