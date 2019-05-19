import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from main import tunning
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


data = np.loadtxt('data-NTE-MnZnN.csv', delimiter=',', skiprows=1)

X = data[:, 1: 6]
scaler_X = StandardScaler()
scaler_X.fit(X)
X_std = scaler_X.transform(X)

y_alpha = data[:, 6]
scaler_y_alpha, y_alpha_std, model_y_alpha = build_model(X_std, y_alpha)
print(MyGridSearch().search(lambda x: -scaler_y_alpha.inverse_transform(model_y_alpha.predict(scaler_X.transform([x]))),
                            [(0, 1) for i in range(5)], [10 for i in range(5)],
                            [lambda x: abs(x[:3].sum() - 1) < 1e-6, lambda x: abs(x[3:].sum() - 1) < 1e-6]))

y_range = data[:, 7]
scaler_y_range, y_range_std, model_y_range = build_model(X_std, y_range)
print(MyGridSearch().search(lambda x: scaler_y_range.inverse_transform(model_y_range.predict(scaler_X.transform([x]))),
                            [(0, 1) for i in range(5)], [10 for i in range(5)],
                            [lambda x: abs(x[:3].sum() - 1) < 1e-6, lambda x: abs(x[3:].sum() - 1) < 1e-6]))

y_start = data[:, 8]
scaler_y_start, y_start_std, model_y_start = build_model(X_std, y_start)

y_end = data[:, 9]
scaler_y_end, y_end_std, model_y_end = build_model(X_std, y_end)

print(MyGridSearch().search(lambda x: scaler_y_end.inverse_transform(model_y_end.predict(scaler_X.transform([x]))) - scaler_y_start.inverse_transform(model_y_start.predict(scaler_X.transform([x]))),
                            [(0, 1) for i in range(5)], [10 for i in range(5)],
                            [lambda x: abs(x[:3].sum() - 1) < 1e-6, lambda x: abs(x[3:].sum() - 1) < 1e-6]))
