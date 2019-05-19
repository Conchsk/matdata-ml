import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture


class MyGaussianMixtureModel:
    def __init__(self):
        pass

    def _tunning(self, X: np.ndarray):
        n_components_try = [i for i in range(1, min(21, X.shape[0] + 1))]
        bic = []
        for n_components in n_components_try:
            model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0).fit(X)
            bic.append(model.bic(X))
        # plt.plot(n_components_try, bic, label='BIC')
        # plt.show()
        return n_components_try[bic.index(min(bic))]

    def fit(self, X: np.ndarray):
        opt_n_components = self._tunning(X)
        self.model = GaussianMixture(n_components=opt_n_components, covariance_type='full', random_state=0).fit(X)

    def sample(self, n_samples: int = 1):
        return self.model.sample(n_samples)[0]


if __name__ == '__main__':
    toy_X, toy_y = make_moons(noise=0.05, random_state=0)
    # plt.scatter(toy_X[:, 0], toy_X[:, 1])
    # plt.show()
    model = MyGaussianMixtureModel()
    model.fit(toy_X)
    sample_X = model.sample(100)
    print(sample_X)
    plt.scatter(toy_X[:, 0], toy_X[:, 1])
    plt.show()
