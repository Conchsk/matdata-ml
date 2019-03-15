import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, linear_model, preprocessing, neighbors, tree
from sklearn.model_selection import LeaveOneOut


def pmo(alg_no, rm_38, type_filter, feature_selection):
    raw_data = np.loadtxt('data-PMO.csv', delimiter=',', skiprows=1)

    if rm_38:
        raw_data = raw_data[np.where(raw_data[:, 12] != 38.98)]

    index = np.array([], dtype=int)
    for it in type_filter:
        index = np.append(index, np.where(raw_data[:, it] == 1))
    raw_data = raw_data[index, :]
    data = preprocessing.StandardScaler().fit(raw_data).transform(raw_data)
    X = data[:, type_filter + feature_selection]
    y = data[:, 12]

    tot = np.linalg.norm(y - np.mean(y)) ** 2

    mse_min = 10000
    hp_opt = 0
    py_opt = []

    for hp in range(1, 2):
        py = []
        ry = []

        loo = LeaveOneOut()
        for train, test in loo.split(X, y):
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]

            if alg_no == 0:
                # Lasso
                model = linear_model.LinearRegression()
            elif alg_no == 1:
                # SVR
                model = svm.SVR()
            elif alg_no == 2:
                # KNR
                model = neighbors.KNeighborsRegressor(hp, weights='distance')
            else:
                # DTR
                model = tree.DecisionTreeRegressor(
                    max_depth=hp, random_state=0)

            model.fit(X_train, y_train)
            py.append(model.predict(X_test))
            ry.append(y_test)

        mse = np.linalg.norm(np.array(py) - np.array(ry)) ** 2
        if mse < mse_min:
            mse_min = mse
            hp_opt = hp
            py_opt = py

    print(1 - mse_min / tot)
    print(hp_opt)
    plt.plot([min(np.min(y), np.min(py_opt)), max(np.max(y), np.max(py_opt))],
             [min(np.min(y), np.min(py_opt)), max(np.max(y), np.max(py_opt))])
    py_opt = np.array(py_opt)
    if 0 in type_filter:
        plt.scatter(y[np.where(raw_data[:, 0] == 1)], py_opt[np.where(
            raw_data[:, 0] == 1)], c='r', label='20CrMnTi')
    if 1 in type_filter:
        plt.scatter(y[np.where(raw_data[:, 1] == 1)],
                    py_opt[np.where(raw_data[:, 1] == 1)], c='g', label='45#')
    if 2 in type_filter:
        plt.scatter(y[np.where(raw_data[:, 2] == 1)], py_opt[np.where(
            raw_data[:, 2] == 1)], c='b', label='60Si2Mn')
    if 3 in type_filter:
        plt.scatter(y[np.where(raw_data[:, 3] == 1)],
                    py_opt[np.where(raw_data[:, 3] == 1)], c='k', label='AM2')
    if 4 in type_filter:
        plt.scatter(y[np.where(raw_data[:, 4] == 1)], py_opt[np.where(
            raw_data[:, 4] == 1)], c='y', label='GCr15')
    if 5 in type_filter:
        plt.scatter(y[np.where(raw_data[:, 5] == 1)], py_opt[np.where(
            raw_data[:, 5] == 1)], c='c', label='SA-210C')
    if 6 in type_filter:
        plt.scatter(y[np.where(raw_data[:, 6] == 1)], py_opt[np.where(
            raw_data[:, 6] == 1)], c='m', label='ZTM-S2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    raw_data = np.loadtxt('data-PMO.csv', delimiter=',', skiprows=1)
