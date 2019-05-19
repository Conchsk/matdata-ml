# coding=utf-8
from bayes_opt import BayesianOptimization, UtilityFunction


def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return x**2 + y**2 + 2*x*y


pbounds = {'x': (0, 1), 'y': (0, 1)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
optimizer.maximize(init_points=2, n_iter=3)
