import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor


class MyBayesianOptimization:
    def __init__(self, bounds: list = [], constraints: list = []):
        self.gpr = GaussianProcessRegressor(random_state=0)
        self.bounds = bounds
        self.constraints = constraints
        self.X = None
        self.y = None
        self.y_max = 0

    def acq(self, X: np.ndarray):
        mean, std = self.gpr.predict(X.reshape(1, -1), return_std=True)
        if std < 1e-6:
            return 0
        else:
            return (mean - self.y_max) * norm.cdf((mean - self.y_max) / std) + std * norm.pdf((mean - self.y_max) / std)

    def suggest(self):
        best_fun = 0.0
        best_result = None
        for i in range(self.X.shape[0]):
            opt_result = minimize(lambda X: -self.acq(X), self.X[i], method='SLSQP',
                                  bounds=self.bounds, constraints=self.constraints)
            if best_fun > opt_result['fun']:
                best_fun = opt_result['fun']
                best_result = opt_result
        return best_result['x']

    def register(self, X: np.ndarray, y: np.ndarray):
        self.X = np.array(X) if self.X is None else np.concatenate((self.X, X))
        self.y = np.array(y) if self.y is None else np.concatenate((self.y, y))
        self.y_max = max(self.y_max, self.y.max())
        self.gpr.fit(self.X, self.y)

    def predict(self, X: np.ndarray, return_std: bool = False):
        return self.gpr.predict(X, return_std=return_std)

    def visualization(self):
        sample_x = np.arange(-2, 10, 0.1)
        print(sample_x)
        mean, std = self.gpr.predict(sample_x.reshape(-1, 1), return_std=True)
        plt.plot(sample_x.reshape(-1, 1), mean, c='r')
        plt.fill_between(sample_x, mean.reshape(-1) - std, mean.reshape(-1) + std)
        plt.show()
