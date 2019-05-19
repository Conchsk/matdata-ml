import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class MySVR:
    def __init__(self):
        self.pca = PCA(n_components=7)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.loor2 = 0

    def _tunning(self, X: np.ndarray, y: np.ndarray, start: int = 1, end: int = 100002, step: int = 10000):
        opt_mse = float('inf')
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
        print(f'opt C is {opt_C / 100} with R^2 = {1 - opt_mse / y.var() }')
        if step == 1:
            return opt_C / 100, opt_mse
        else:
            return self._tunning(X, y, max(opt_C - step, start), min(opt_C + step, end), step // 10)

    def fit(self, X, y):
        self.pca.fit(X)
        new_X = self.pca.transform(X)
        # new_X = X
        self.scaler_X.fit(new_X)
        self.scaler_y.fit(y.reshape(-1, 1))
        X_std = self.scaler_X.transform(new_X)
        y_std = self.scaler_y.transform(y.reshape(-1, 1))
        opt_C, opt_mse = self._tunning(X_std, y_std.reshape(-1))
        self.model = SVR(C=opt_C, gamma='scale')
        self.model.fit(X_std, y_std.reshape(-1))
        # self.loor2 = 1 - opt_mse / np.linalg.norm(y_std - y_std.mean())**2

    def predict(self, X):
        X_std = self.scaler_X.transform(X)
        return self.scaler_y.inverse_transform(self.model.predict(X_std))

    def score(self, X, y):
        X_std = self.scaler_X.transform(X)
        y_std = self.scaler_y.transform(y.reshape(-1, 1))
        return self.model.score(X_std, y_std.reshape(-1))

    def mse(self, X, y):
        X_std = self.scaler_X.transform(X)
        y_std = self.scaler_y.transform(y.reshape(-1, 1))
        y_pred = self.model.predict(X_std)
        return np.linalg.norm(y_pred - y_std.reshape(-1))**2 / y_pred.shape[0]
