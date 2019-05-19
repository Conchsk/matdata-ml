import numpy as np

from MyBayesianOptimization import MyBayesianOptimization

data = np.loadtxt('data-NTE-final.csv', delimiter=',', skiprows=1)

X = data[:, 1:8]
y = 0 - data[:, 8]
bys_opt = MyBayesianOptimization(bounds=[(0, 1) for i in range(7)], constraints=[lambda x: np.sum(x) - 1])
bys_opt.register(X, y)
opt_X = bys_opt.suggest()
print(opt_X)
print(bys_opt.predict([opt_X], return_std=True))
