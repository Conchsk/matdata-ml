import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm, linear_model

data = np.loadtxt('data-GCr15.csv', delimiter=',', skiprows=1)

rm_38 = True
if rm_38:
    data = data[np.where(data[:, 5] != 38.98)]

only_openm = False
if only_openm:
    data = data[np.where(data[:, 4] == 1)]

feature_select = [0, 1, 2, 3, 4]
data = data[:, feature_select + [-2]]

X = data[:, :-1]
scalerX = preprocessing.StandardScaler()
scalerX.fit(X)
standardX = True
if standardX:
    X = scalerX.transform(X)

y = data[:, -1]
scalery = preprocessing.StandardScaler()
scalery.fit(y.reshape((-1, 1)))
standardy = True
if standardy:
    y = scalery.transform(y.reshape((-1, 1))).reshape(-1)

model = linear_model.Lasso(alpha=0.05)
# model = svm.SVR(kernel='linear')
model.fit(X, y)
py = model.predict(X)
tot = np.linalg.norm(y - np.mean(y)) ** 2
mse = np.linalg.norm(y - py) ** 2
print(1 - mse / tot)
plt.plot([min(np.min(y), np.min(py)), max(np.max(y), np.max(py))],
         [min(np.min(y), np.min(py)), max(np.max(y), np.max(py))])
plt.scatter(y, py)
plt.show()

# if 0 in feature_select:
#     V = [i / 100 for i in range(50, 100)]
# else:
#     V = [0]
#
# if 1 in feature_select:
#     T = [i for i in range(20, 50)]
# else:
#     T = [0]
#
# if 2 in feature_select:
#     I = [i for i in range(500, 1000, 10)]
# else:
#     I = [0]
#
# if 3 in feature_select:
#     F = [i for i in range(0, 100, 5)]
# else:
#     F = [0]
#
# if 4 in feature_select:
#     M = [0, 1]
# else:
#     M = [0]
#
# result_max = 0
# result_max_input = None
#
# for i in range(len(V)):
#     for j in range(len(T)):
#         for k in range(len(I)):
#             for l in range(len(F)):
#                 for m in range(len(M)):
#                     tx = np.array([[V[i], T[j], I[k], F[l], M[m]]])[:, feature_select]
#                     if standardX:
#                         tx = scalerX.transform(tx)
#                     result = model.predict(tx)
#                     if result > result_max:
#                         result_max = result
#                         result_max_input = tx
# if standardX:
#     print(scalerX.inverse_transform(result_max_input))
# else:
#     print(result_max_input)
#
# if standardy:
#     print(scalery.inverse_transform(result_max))
# else:
#     print(result_max)
