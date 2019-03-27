import numpy as np


class MyGridSearch:
    def fold_generator(self, folds: list):
        count = [0 for i in range(len(folds))]
        while True:
            for i in range(len(count) - 1, 0, -1):
                count[i - 1] += count[i] // (folds[i] + 1)
                count[i] = count[i] % (folds[i] + 1)
            if count[0] <= folds[0]:
                yield count
            else:
                raise StopIteration()
            count[-1] += 1

    def search(self, func, bounds: list, folds: list, constraints: list):
        opt_X = None
        opt_y = 0
        bounds_arr = np.array(bounds)
        folds_arr = np.array(folds)
        for fold in self.fold_generator(folds):
            x_try = bounds_arr[:, 0] + (bounds_arr[:, 1] - bounds_arr[:, 0]) / folds_arr * np.array(fold)
            check = True
            for constraint in constraints:
                if not constraint(x_try):
                    check = False
                    break
            if check:
                print(x_try)
                y_hat = func(x_try)
                if opt_y < y_hat:
                    opt_y = y_hat
                    opt_X = x_try
        return opt_X, opt_y
