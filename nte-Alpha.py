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


data = np.loadtxt('data-NTE-Alpha.csv', delimiter=',', skiprows=1)

X = data[:, 1: 11]
scaler_X = StandardScaler()
scaler_X.fit(X)
X_std = scaler_X.transform(X)

y_alpha = data[:, 11]
scaler_y_alpha, y_alpha_std, model_y_alpha = build_model(X_std, y_alpha)
print(MyGridSearch().search(lambda x: -scaler_y_alpha.inverse_transform(model_y_alpha.predict(scaler_X.transform([x]))),
                            [(0, 1) for i in range(10)], [4 for i in range(10)],
                            [lambda x: abs(x[:8].sum() - 1) < 1e-6, lambda x: abs(x[8:].sum() - 1) < 1e-6]))
