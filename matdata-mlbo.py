# coding=utf-8
from bayes_opt import BayesianOptimization, UtilityFunction



def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1


pbounds = {'x': (2, 4), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
# next_point_to_probe = optimizer.suggest(utility)
# print("Next point to probe is:", next_point_to_probe)

# target = black_box_function(**next_point_to_probe)
# print("Found the target value to be:", target)

optimizer.register(
    params={'x': 2.8, 'y': 1.3},
    target=-7,
)

next_point_to_probe = optimizer.suggest(utility)
print("Next point to probe is:", next_point_to_probe)
