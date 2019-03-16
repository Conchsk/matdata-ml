import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVC


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
#     'index_removed': [4, 6, 22, 23, 24],
#     'X_index': [i for i in range(3, 14)],
#     'y_index': [14],
# }

# NTE T
config = {
    'csv': 'data-NTE.csv',
    'index_removed': [3, 4, 20, 21],
    'X_index': [i for i in range(3, 13)],
    'y_index': [17],
}


def tunning(X, y):
    opt_C = 0
    min_mse = 10000000
    for it_C in range(1, 20001, 10):
        model = SVR(C=it_C / 100, gamma='auto')
        it_mse = 0
        for train, test in LeaveOneOut().split(X):
            model.fit(X[train], y[train].reshape(-1))
            it_mse += (model.predict(X[test])[0] - y[test].reshape(-1)[0])**2
        if min_mse > it_mse:
            min_mse = it_mse
            opt_C = it_C / 100
    print(f'optimal C is {opt_C} with R^2 = {1 - min_mse / np.linalg.norm(y - y.mean())**2}')
    return opt_C


if __name__ == '__main__':
    raw_data = np.loadtxt(config['csv'], delimiter=',', skiprows=1)
    print(raw_data)
    print('--------------------\n')

    filtered_data = raw_data[[i for i in range(raw_data.shape[0]) if i not in config['index_removed']]]
    print(filtered_data)
    print('--------------------\n')

    X = filtered_data[:, config['X_index']]
    X = StandardScaler().fit_transform(X)
    print(X)
    print('--------------------\n')

    y = filtered_data[:, config['y_index']]
    print(y)
    print('--------------------\n')

    opt_C = tunning(X, y)
    model = SVR(C=opt_C, gamma='auto')
    model.fit(X, y.reshape(-1))
    print(model.score(X, y.reshape(-1)))


# 10 4061/0.43806226031254236
# 11 19951/0.557826453682837
# 12 19831/0.5425249017595752
# 13 4631/0.044084679151381034
