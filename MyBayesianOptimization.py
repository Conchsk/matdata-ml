import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import savefig
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor


class MyBayesianOptimization:
    def __init__(self, bounds: list = [], constraints: list = [], ee_ration: float = 0):
        self.gpr = GaussianProcessRegressor(alpha=1e-6, random_state=0)
        self.bounds = np.array(bounds)
        self.constraints = constraints
        self.ee_ratio = ee_ration
        self.X = None
        self.y = None
        self.opt_y = 0

    def acq(self, X: np.ndarray):
        mean, std = self.gpr.predict(X, return_std=True)
        mean = mean.reshape(-1)
        if std.all() < 1e-6:
            ei = np.zeros(mean.shape)
        else:
            ei = (mean - self.opt_y - self.ee_ratio) * norm.cdf((mean - self.opt_y - self.ee_ratio) / std) + \
                std * norm.pdf((mean - self.opt_y - self.ee_ratio) / std)
        for constraint in self.constraints:
            ei -= 10000 * constraint(X)**2
        return ei

    def acq_ucb(self, X: np.ndarray):
        mean, std = self.gpr.predict(X, return_std=True)
        mean = mean.reshape(-1)
        if std.all() < 1e-6:
            ucb = 0
        else:
            ucb = mean + np.sqrt(self.ee_ratio) * std
        for constraint in self.constraints:
            ucb -= 10000 * constraint(X)**2
        return ucb

    def acq_pi(self, X: np.ndarray):
        mean, std = self.gpr.predict(X, return_std=True)
        mean = mean.reshape(-1)
        if std.all() < 1e-6:
            pi = 0
        else:
            pi = norm.cdf((mean - self.opt_y - self.ee_ratio) / std)
        for constraint in self.constraints:
            pi -= 10000 * constraint(X)**2
        return pi

    def suggest(self):
        best_fun = 12345678
        best_result = None

        init_points = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=(10, self.bounds.shape[0]))
        for init_point in init_points:
            opt_result = minimize(lambda X: -self.acq([X]), init_point, bounds=self.bounds)
            if best_fun > opt_result['fun']:
                best_fun = opt_result['fun']
                best_result = opt_result

        return best_result['x']

    def register(self, X: np.ndarray, y: np.ndarray):
        self.X = np.array(X) if self.X is None else np.concatenate((self.X, X))
        self.y = np.array(y) if self.y is None else np.concatenate((self.y, y))
        self.opt_y = max(self.opt_y, self.y.max())
        self.gpr.fit(self.X, self.y)

    def predict(self, X: np.ndarray, return_std: bool = False):
        return self.gpr.predict(X, return_std=return_std)

    class UpdatePlot:
        def __init__(self, ax):
            self.line, = ax.plot([], [], 'k-')
            self.x = np.arange(-2, 10 + 0.1, 0.1)            
            ax.set_xlim(-2, 10)
            ax.set_ylim(0, 1)

        def init(self):
            self.line.set_data([], [])
            return self.line,

        def __call__(self, i):
            if i == 0:
                return self.init()
            
            y = np.random.uniform(size=self.x.shape)
            self.line.set_data(self.x, y)
            return self.line,

    def visualization_2d(self, real_func: callable, i):
        # X = np.arange(self.bounds[0][0], self.bounds[0][1] + 0.1, 0.1)
        # y_real = real_func(X)
        # y_mean, y_std = self.gpr.predict(X.reshape(-1, 1), return_std=True)
        # plt.figure(figsize=(6, 2))
        # plt.plot(X, y_real, color='r', label="Real")
        # plt.plot(X, y_mean.reshape(-1), color='k', label="BO")
        # plt.fill_between(X, y_mean.reshape(-1) - y_std, y_mean.reshape(-1) + y_std, color='g', alpha=0.5)
        # plt.legend()
        # plt.show()
        # savefig(f'D:/test_{i}.png')

        fig, ax = plt.subplots()
        up = MyBayesianOptimization.UpdatePlot(ax)
        anim = FuncAnimation(fig, up, frames=np.arange(100), init_func=up.init, interval=100, blit=True)
        plt.show()

    def visualization_3d(self):
        X = np.arange(self.bounds[0][0], self.bounds[0][1], 0.1)
        Y = np.arange(self.bounds[1][0], self.bounds[1][1], 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = self.gpr.predict(np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)).reshape(X.shape)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
