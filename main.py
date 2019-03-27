import matplotlib.pyplot as plt
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

config = {
    'csv': 'paper-train.csv',
    'index_removed': [],
    # 'index_removed': [2, 3, 4, 16, 18, 19, 20, 21, 22, 23],
    'X_index': [i for i in range(1, 11)],
    'y_index': [11]
}


def feature_visualization(X: np.ndarray, y: np.ndarray):
    shape = [(1, 1), (1, 2), (2, 2), (2, 2), (2, 3), (2, 3), (3, 3), (3, 3), (3, 3), (3, 4), (3, 4), (3, 4)]
    print(X.shape[1])
    if X.shape[1] > len(shape):
        print('too many features')
        return
    else:
        ret = plt.subplots(shape[X.shape[1]][0], shape[X.shape[1]][1])
        for index in range(X.shape[1]):
            ret[1][index // shape[X.shape[1]][1], index % shape[X.shape[1]][1]].scatter(X[:, index], y)
        plt.show()


def norm_2d(X1, X2, mu1: float = 0, mu2: float = 0, sigma1: float = 1, sigma2: float = 1, rho: float = 0):
    Z1 = (X1 - mu1)**2 / sigma1**2
    Z2 = (X2 - mu2)**2 / sigma2**2
    Z12 = 2 * rho * (X1 - mu1) * (X2 - mu2) / (sigma1 * sigma2)
    return 1 / (2 * np.pi * sigma1 * sigma2 * np.sqrt(1 - rho**2)) * np.exp(-1 / (2 * (1 - rho**2)) * (Z1 + Z2 - Z12))


def plot_3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(-2, 2, 0.1)
    Y = np.arange(-2, 2, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = norm_2d(X, Y, mu1=1)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def tunning(X: np.ndarray, y: np.ndarray, start: int = 1, end: int = 100002, step: int = 10000):
    opt_mse = 10000000
    opt_kernel = 'rbf'
    opt_C = 0
    for C in range(start, end, step):
        for kernel in ['rbf']:
            model = SVR(kernel=kernel, C=C / 100, gamma='auto')
            mse = 0
            for train, test in LeaveOneOut().split(X):
                model.fit(X[train], y[train].reshape(-1))
                mse += (model.predict(X[test])[0] - y[test].reshape(-1)[0])**2
            if opt_mse > mse:
                opt_mse = mse
                opt_kernel = kernel
                opt_C = C
    print(f'opt kernel is {opt_kernel}, opt C is {opt_C / 100} '
          f'with R^2 = {1 - opt_mse / np.linalg.norm(y - y.mean())**2}')
    if step == 1:
        return opt_kernel, opt_C / 100
    else:
        return tunning(X, y, opt_C - step, opt_C + step, int(step / 10))


def visualization_3d(X_bound: tuple, Y_bound: tuple, Z_func: callable):
    X = np.arange(X_bound[0], X_bound[1], (X_bound[1] - X_bound[0]) / 100)
    Y = np.arange(Y_bound[0], Y_bound[1], (Y_bound[1] - Y_bound[0]) / 100)
    X, Y = np.meshgrid(X, Y)
    Z = Z_func(X.reshape(-1, 1), Y.reshape(-1, 1)).reshape(X.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    # raw_data = np.loadtxt(config['csv'], delimiter=',', skiprows=1)
    # print(raw_data)
    # print('--------------------\n')

    # filtered_data = raw_data[[i for i in range(raw_data.shape[0]) if i not in config['index_removed']]]
    # print(filtered_data)
    # print('--------------------\n')

    # X = filtered_data[:, config['X_index']]
    # scaler = StandardScaler()
    # scaler.fit(X)
    # std_X = scaler.transform(X)
    # print(X)
    # print('--------------------\n')

    # y = filtered_data[:, config['y_index']]
    # print(y)
    # print('--------------------\n')

    # # feature_visualization(X, y)

    # opt_kernel, opt_C = tunning(std_X, y)
    # model = SVR(kernel=opt_kernel, C=opt_C, gamma='auto')
    # model.fit(std_X, y.reshape(-1))
    # print(model.score(std_X, y.reshape(-1)))

    # bys_opt = MyBayesianOptimization(bounds=[(-1, 1) for i in range(X.shape[1])],
    #                                  constraints=[{'type': 'eq', 'fun': lambda X: X[:8].sum() - 1},
    #                                               {'type': 'eq', 'fun': lambda X: X[8:].sum() - 1}])
    # bys_opt.register(X, y)
    # opt_x = bys_opt.suggest()
    # mean, std = bys_opt.predict([opt_x], return_std=True)
    # print(f'opt_x is \n{opt_x} \nwith mean = {mean[0][0]}, std = {std[0]}')
    # print('--------------------\n')
    # def func_test(x):
    #     return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1 / (x**2 + 1)

    # bys_opt = MyBayesianOptimization(bounds=[(-3, 3), (-3, 3)], constraints=[{'type': 'ineq', 'fun': lambda x: 1 - (x[0]**2 + x[1]**2)}])
    # bys_opt.register([[0, 0]], [[norm_2d(0, 0)]])
    # # bys_opt.register([[0.5, 0.5]], [[norm_2d(0.5, 0.5)]])
    # # bys_opt.register([[-0.5, -0.5]], [[norm_2d(-0.5, -0.5)]])
    # for i in range(10):
    #     opt_x = bys_opt.suggest()
    #     print(opt_x)
    #     y_hat = norm_2d(opt_x[0], opt_x[1], mu1=1)
    #     bys_opt.register([opt_x], [[y_hat]])
    #     # print(f'current registered point is X = \n{opt_x}\n y_hat = {y_hat}')
    #     # bys_opt.visualization()
    # # plot_3d()
    # pass

    def test_func(x, y):
        return x*y

    gpr = GaussianProcessRegressor(random_state=0)
    gpr.fit([[0.5, 0], [-1, 0], [-0.5, 0.5]], [test_func(0.5, 0), test_func(-1, 0), test_func(-0.5, 0.5)])
    visualization_3d((0, 1), (0, 1), lambda X, Y: gpr.predict(np.concatenate((X, Y), axis=1)))
    # plot_3d()
    # opt_result = minimize(lambda x: -norm_2d(x[0], x[1], mu1=1), [0, 0], bounds=[(-1, 1), (-1, 1)],
    #                       constraints=[{'type': 'ineq', 'fun': lambda x: 1 - (x[0]**2 + x[1]**2)}])
    # print(opt_result)
