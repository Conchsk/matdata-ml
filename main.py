import numpy as np
from matplotlib import cm
from matplotlib.pyplot import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from MyBayesianOptimization import MyBayesianOptimization
from MyGaussianMixtureModel import MyGaussianMixtureModel
from MySVR import MySVR


# def feature_visualization(X: np.ndarray, y: np.ndarray):
#     shape = [(1, 1), (1, 2), (2, 2), (2, 2), (2, 3), (2, 3), (3, 3), (3, 3), (3, 3), (3, 4), (3, 4), (3, 4)]
#     print(X.shape[1])
#     if X.shape[1] > len(shape):
#         print('too many features')
#         return
#     else:
#         ret = plt.subplots(shape[X.shape[1]][0], shape[X.shape[1]][1])
#         for index in range(X.shape[1]):
#             ret[1][index // shape[X.shape[1]][1], index % shape[X.shape[1]][1]].scatter(X[:, index], y)
#         plt.show()


def tunning(X: np.ndarray, y: np.ndarray, start: int = 1, end: int = 1000002, step: int = 100000):
    opt_mse = 10000000
    opt_C = 0
    for C in range(start, end, step):
        model = SVR(C=C / 100, gamma='scale')
        mse = 0
        for train, test in LeaveOneOut().split(X):
            model.fit(X[train], y[train].reshape(-1))
            mse += (model.predict(X[test])[0] - y[test].reshape(-1)[0])**2
        if opt_mse > mse:
            opt_mse = mse
            opt_C = C
    print(f'opt C is {opt_C / 100} with R^2 = {1 - opt_mse / np.linalg.norm(y - y.mean())**2}')
    if step == 1:
        return opt_C / 100
    else:
        return tunning(X, y, opt_C - step, opt_C + step, int(step / 10))


class DataAdapter:
    def __init__(self):
        self.raw_data = np.loadtxt('data-NTE2.csv', delimiter=',', skiprows=1)[:, :19]

    def _random_split(self, X: np.ndarray, factor: float):
        flag_arr = [False for i in range(X.shape[0])]
        true_num = 0
        while true_num < len(flag_arr) * factor:
            next_int = np.random.randint(0, len(flag_arr))
            if not flag_arr[next_int]:
                flag_arr[next_int] = True
                true_num += 1
        train_index = np.where(np.array(flag_arr) == True)
        test_index = np.where(np.array(flag_arr) == False)
        return X[train_index], X[test_index]

    def get_data(self, factor: float, useGMM: bool = False, n_samples: int = 1, append: bool = False):
        train, test = self._random_split(self.raw_data, factor)
        if useGMM:
            model = MyGaussianMixtureModel()
            model.fit(train)
            if append:
                train = np.concatenate((train, model.sample(n_samples)))
            else:
                train = model.sample(n_samples)
        return {'X': train[:, :15], 'y': train[:, 15]}, {'X': test[:, :15], 'y': test[:, 15]}


if __name__ == '__main__':
    with open('result.csv', 'w+') as fp:
        for seed in range(20):
            for factor in range(2, 9):
                # data
                da = DataAdapter()
                np.random.seed(seed)
                train, test = da.get_data(factor / 10)

                # train
                model = MySVR()
                model.fit(train['X'], train['y'].reshape(-1, 1))
                fp.write(str(round(model.loor2, 2)))
                fp.write(',')

                # test
                fp.write(str(round(model.score(test['X'], test['y'].reshape(-1, 1)), 2)))
                fp.write(',')

            for factor in range(2, 9):
                # data
                da = DataAdapter()
                np.random.seed(seed)
                train, test = da.get_data(factor / 10, useGMM=True, n_samples=int(factor), append=True)

                # train
                model = MySVR()
                model.fit(train['X'], train['y'].reshape(-1, 1))
                fp.write(str(round(model.loor2, 2)))
                fp.write(',')

                # test
                fp.write(str(round(model.score(test['X'], test['y'].reshape(-1, 1)), 2)))
                fp.write(',')
            
            for factor in range(2, 9):
                # data
                da = DataAdapter()
                np.random.seed(seed)
                train, test = da.get_data(factor / 10, useGMM=True, n_samples=int(4 * factor), append=True)

                # train
                model = MySVR()
                model.fit(train['X'], train['y'].reshape(-1, 1))
                fp.write(str(round(model.loor2, 2)))
                fp.write(',')

                # test
                fp.write(str(round(model.score(test['X'], test['y'].reshape(-1, 1)), 2)))
                fp.write(',')

            for factor in range(2, 9):
                # data
                da = DataAdapter()
                np.random.seed(seed)
                train, test = da.get_data(factor / 10, useGMM=True, n_samples=int(2 * factor), append=True)

                # train
                model = MySVR()
                model.fit(train['X'], train['y'].reshape(-1, 1))
                fp.write(str(round(model.loor2, 2)))
                fp.write(',')

                # test
                fp.write(str(round(model.score(test['X'], test['y'].reshape(-1, 1)), 2)))
                if factor < 8:
                    fp.write(',')
                else:
                    fp.write('\n')
