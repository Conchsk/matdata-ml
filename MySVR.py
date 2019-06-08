import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR


class MySVR:
    def __init__(self, kernel='rbf'):
        self.model = make_pipeline(StandardScaler(), SVR(kernel=kernel, cache_size=1024))

    def _tunning(self, X, y, start=0, end=16, step=4):
        opt_score = -float('inf')
        opt_C = 0
        for cur_C in range(start, end + step, step):
            self.model.set_params(svr__C=2**cur_C)
            scores = cross_val_score(self.model, X, y, cv=3)
            if opt_score < np.mean(scores):
                opt_score = np.mean(scores)
                opt_C = cur_C
        print(opt_score, 2**opt_C)
        if step == 1:
            self.model.set_params(svr__C=2**opt_C)
        else:
            self._tunning(X, y, max(opt_C - step, start), min(opt_C + step, end), step // 2)

    def fit(self, X, y):
        # self._tunning(X, y)
        self.model.set_params(svr__C=100000)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def visualize(self, X, y):
        y_hat = self.predict(X)
        plt.scatter(y, y_hat, c='r', marker='.')
        left_bottom = min(y.min(), y_hat.min()) * 0.9
        right_top = max(y.max(), y_hat.max()) * 1.1
        plt.plot([left_bottom, right_top], [left_bottom, right_top], c='g')
        plt.xlabel('y_real')
        plt.ylabel('y_pred')
        plt.show()


if __name__ == '__main__':
    raw_data = np.loadtxt('data-SNN.csv', delimiter=',', skiprows=1)
    X = raw_data[:, :5]
    y = raw_data[:, 5]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1234)

    model = MySVR()
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))
    y_hat = model.predict(X_test)
    statistics = np.abs((y_hat - y_test) / y_test)
    print('<10%', (statistics < 0.1).sum())
    print('>10% && <50%', ((statistics > 0.1) & (statistics < 0.5)).sum())
    print('>50%', (statistics > 0.5).sum())

    model.visualize(X_test, y_test)
