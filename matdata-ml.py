import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

config = {
    "PMO-GCr15": {
        "csv": "data-PMO-GCr15.csv",
        "filter": [6, 7, 10],
        "X": [0, 1, 2, 3],
        "y": [5],
        "bo": {
            "pbounds": {"V": (0.69, 0.82), "T": (22, 41), "I": (390, 942), "F": (55, 115)}
        }
    },
    "NTE": {
        "csv": "data-NTE.csv",
        "filter": [0, 3],
        "X": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "y": [16],
        "bo": {
            "pbounds": {"Mn": (0, 1), "Fe": (0, 1),
                        "Cu": (0, 1), "Ge": (0, 1), "Ga": (0, 1), "Mn2": (0, 1), "Zn": (0, 1), "Si": (0, 1), "Sn": (0, 1), "Nb": (0, 1),
                        "N": (0, 1), "C": (0, 1), "B": (0, 1)}
        }
    },
    "NTE-MnZnN": {
        "csv": "data-NTE-MnZnN.csv",
        "filter": [],
        "X": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "y": [9]
    },
    "NTE-min": {
        "csv": "data-NTE-min.csv",
        "filter": [],
        "X": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "y": [13]
    },
    "NTE-plus": {
        "csv": "data-NTE-plus.csv",
        "filter": [],
        "X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "y": [11]
    }
}

atomic_radius = np.diag([0.128, 0.123, 0.122, 0.124, 0.133,
                         0.117, 0.141, 0.143, 0.071, 0.0771])
electronegativity = np.diag(
    [1.9, 2.01, 1.81, 1.55, 1.65, 1.9, 1.96, 1.6, 3.04, 2.55])
valence_electron = np.diag([2, 4, 3, 4, 2, 4, 4, 5, 3, 4])

cur_cfg = config['NTE-plus']
cur_alg = 'svr'

# read csv headers (for bo use)
print('####################')
with open(cur_cfg['csv'], 'r') as fp:
    headers = fp.readline()[:-1]
headers = headers.split(',')
print('headers:', headers)

# read csv as raw data
print('####################')
raw_data = np.loadtxt(cur_cfg['csv'], delimiter=',', skiprows=1)
print('raw data:', raw_data)

# filter instances
print('####################')
filter_arr = np.zeros(raw_data.shape[0])
filter_arr[cur_cfg['filter']] = 1
filtered_data = raw_data[np.where(filter_arr == 0)]
print('filtered data:', filtered_data)

# select features
print('####################')
X = filtered_data[:, cur_cfg['X']]
# ar_feature = np.matmul(X, atomic_radius)
# en_feature = np.matmul(X, electronegativity)
# ve_feature = np.matmul(X, valence_electron)
# X = np.hstack((X, ar_feature))
# X = np.hstack((X, en_feature))
# X = np.hstack((X, ve_feature))
if cur_alg == 'svr':
    std = StandardScaler()
    X = std.fit_transform(X)
print('X:', X)

# select label(s)
print('####################')
y = filtered_data[:, cur_cfg['y']]
print('y:', y)

# select algorithm & tunning
print('####################')
if cur_alg == 'lasso':
    model = Lasso(alpha=0.5, normalize=True)
elif cur_alg == 'random-forest':
    model = RandomForestRegressor(
        max_depth=2, random_state=0, n_estimators=100)
elif cur_alg == 'svr':
    best_C = 0
    best_mse = 1000
    for it_C in range(1, 10000, 10):
        model = SVR(C=it_C / 100, gamma='auto')
        loo = LeaveOneOut()
        it_mse = 0.0
        for train, test in loo.split(X):
            it_mse += (model.fit(X=X[train], y=y[train].reshape(-1)
                                 ).predict(X[test]) - y[test][0])**2
        if best_mse > it_mse:
            best_mse = it_mse
            best_C = it_C
    model = SVR(C=best_C / 100, gamma='auto')
elif cur_alg == 'bo':
    pbounds = {}
    for i in cur_cfg['X']:
        pbounds.setdefault(headers[i], cur_cfg['bo']['pbounds'][headers[i]])
    print(pbounds)
    model = BayesianOptimization(
        f=None, pbounds=pbounds, verbose=2, random_state=1)
else:
    model = None

# train & validate
print('####################')
if cur_alg in ['lasso', 'random-forest', 'svr']:
    model.fit(X=X, y=y.reshape(-1))
    print(model.score(X=X, y=y.reshape(-1)))

    loo = LeaveOneOut()
    res_square = 0.0
    for train, test in loo.split(X):
        res_square += (model.fit(X=X[train], y=y[train].reshape(-1)
                                 ).predict(X[test]) - y[test][0])**2
    r_square = 1.0 - res_square / np.linalg.norm(y - y.mean())**2
    print(r_square)
elif cur_alg == 'bo':
    for i in range(y.shape[0]):
        params = {}
        for j in range(len(cur_cfg['X'])):
            params.setdefault(headers[cur_cfg['X'][j]], X[i][j])
        model.register(params=params, target=y[i][0])
else:
    pass

# predict
print('####################')
if cur_alg in ['ols', 'svr']:
    pass
    # print(model.predict(X=[[0.11, 0.021, 0.12, 0.11, 0.20, 0.27, 0.07, 0.1, 1]]))
    pass
elif cur_alg == 'bo':
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    next_point_to_probe = model.suggest(utility)
    print('next point to probe:', next_point_to_probe)
else:
    pass
