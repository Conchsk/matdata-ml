import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from main import tunning
from MyBayesianOptimization import MyBayesianOptimization
from MyGridSearch import MyGridSearch


def build_model(X_std, y):
    scaler = StandardScaler()
    scaler.fit(y.reshape(-1, 1))
    y_std = scaler.transform(y.reshape(-1, 1)).reshape(-1)

    opt_C = tunning(X_std, y_std)
    model = SVR(C=opt_C, gamma='scale')
    model.fit(X_std, y_std)
    print(model.score(X_std, y_std))

    return scaler, y_std, model


data = np.loadtxt('data-NTE-temper.csv', delimiter=',', skiprows=1)

X = data[:, 1: 11]
scaler_X = StandardScaler()
scaler_X.fit(X)
X_std = scaler_X.transform(X)

y_range = data[:, 11]
# scaler_y_range, y_range_std, model_y_range = build_model(X_std, y_range)
# print(MyGridSearch().search(lambda x: scaler_y_range.inverse_transform(model_y_range.predict(scaler_X.transform([x]))),
#                             [(0, 1) for i in range(10)], [4 for i in range(10)],
#                             [lambda x: abs(x[:8].sum() - 1) < 1e-6, lambda x: abs(x[8:].sum() - 1) < 1e-6]))

# y_start = data[:, 12]
# scaler_y_start, y_start_std, model_y_start = build_model(X_std, y_start)

# y_end = data[:, 13]
# scaler_y_end, y_end_std, model_y_end = build_model(X_std, y_end)

# print(MyGridSearch().search(lambda x: scaler_y_end.inverse_transform(model_y_end.predict(scaler_X.transform([x]))) - scaler_y_start.inverse_transform(model_y_start.predict(scaler_X.transform([x]))),
#                             [(0, 1) for i in range(10)], [4 for i in range(10)],
#                             [lambda x: abs(x[:8].sum() - 1) < 1e-6, lambda x: abs(x[8:].sum() - 1) < 1e-6]))

bys_opt = MyBayesianOptimization(bounds=[(0, 1) for i in range(10)],
                                 constraints=[lambda x: x[:8].sum() - 1, lambda x: x[8:].sum() - 1])
# bys_opt.register(X[:, [0, 1, 2, 3, 4, 5, 6, 8]], y_range)

def test_func(X):
    return scaler_y_end.inverse_transform(model_y_end.predict(scaler_X.transform([X])))[0] - \
           scaler_y_start.inverse_transform(model_y_start.predict(scaler_X.transform([X])))[0]

# bys_opt.register([[1,0,0,0,0,0,0,1]], [test_func([1,0,0,0,0,0,0,0,1,0])])
# opt_X = None
# opt_y = 0
# for i in range(100):
#     print(i)
#     next_X = bys_opt.suggest()
#     next_real_X = next_X[:7].tolist()
#     next_real_X.append(max(0, 1 - next_X[:7].sum()))
#     next_real_X.append(next_X[7])
#     next_real_X.append(max(0, 1 - next_X[7]))
#     next_y = test_func(next_real_X)
#     bys_opt.register([next_X], [next_y])
#     if next_y > opt_y:
#         opt_y = next_y
#         opt_X = next_X
# print(opt_X)
# print(opt_y)

bys_opt.register(X, y_range)
opt_X = bys_opt.suggest()
print(opt_X)
print(bys_opt.predict([opt_X], return_std=True))
