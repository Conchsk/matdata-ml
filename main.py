import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVC
import matplotlib.pyplot as plt

# Test
# config = {
#     'csv': 'data-Test.csv',
#     'index_removed': [],
#     'X_index': [i for i in range(0, 2)],
#     'y_index': [2],
# }

# NTE alpha
# config = {
#     'csv': 'data-NTE.csv',
#     'index_removed': [4, 6, 19, 20, 21, 22, 23],
#     'X_index': [i for i in range(4, 12)],
#     'y_index': [14],
# }

# NTE T
config = {
    'csv': 'data-NTE.csv',
    'index_removed': [3, 4, 19, 20],
    'X_index': [i for i in range(3, 13)],
    'y_index': [17],
}


def feature_visualization(X: np.ndarray, y: np.ndarray):
    shape = [(1, 1), (1, 2), (2, 2), (2, 2), (2, 3), (2, 3), (3, 3), (3, 3), (3, 3), (3, 4), (3, 4), (3, 4)]
    print(X.shape[1])
    if X.shape[1] > len(shape):
        print('too many features')
        return
    else:
        f, ax = plt.subplots(shape[X.shape[1]][0], shape[X.shape[1]][1])
        for index in range(X.shape[1]):
            ax[index // shape[X.shape[1]][1], index % shape[X.shape[1]][1]].scatter(X[:, index], y)
        plt.show()


def tunning(X: np.ndarray, y: np.ndarray, start: int = 1, end: int = 100002, step: int = 1000):
    opt_C = 0
    min_mse = 10000000
    for it_C in range(start, end, step):
        model = SVR(C=it_C / 100, gamma='auto')
        it_mse = 0
        for train, test in LeaveOneOut().split(X):
            model.fit(X[train], y[train].reshape(-1))
            it_mse += (model.predict(X[test])[0] - y[test].reshape(-1)[0])**2
        if min_mse > it_mse:
            min_mse = it_mse
            opt_C = it_C
    if step == 1:
        print(f'optimal C is {opt_C / 100} with R^2 = {1 - min_mse / np.linalg.norm(y - y.mean())**2}')
        return opt_C / 100
    else:
        return tunning(X, y, opt_C - step, opt_C + step, int(step / 10))


if __name__ == '__main__':
    raw_data = np.loadtxt(config['csv'], delimiter=',', skiprows=1)
    print(raw_data)
    print('--------------------\n')

    filtered_data = raw_data[[i for i in range(raw_data.shape[0]) if i not in config['index_removed']]]
    print(filtered_data)
    print('--------------------\n')

    X = filtered_data[:, config['X_index']]
    # X = StandardScaler().fit_transform(X)
    print(X)
    print('--------------------\n')

    y = filtered_data[:, config['y_index']]
    print(y)
    print('--------------------\n')

    feature_visualization(X, y)

    # opt_C = tunning(X, y)
    # model = SVR(C=opt_C, gamma='auto')
    # model.fit(X, y.reshape(-1))
    # print(model.score(X, y.reshape(-1)))
