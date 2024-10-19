from typing import Callable, Dict, List, Tuple, Union

from hyperopt import hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import (
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from xgboost import Booster


def build_hyperparameters_space(
    model_class: Callable[
        ...,
        Union[
            Ridge,
            LinearRegression,
            SVR,
            RandomForestRegressor,
            Booster,
        ],
    ],
    random_state: int = 42,
    **kwargs,
) -> Tuple[Dict, Dict[str, List]]:
    params = {}
    choices = {}

    if SVR is model_class:
        params = {
            'kernel': hp.choice('kernel', ['rbf', 'sigmoid']),
            'gamma': hp.loguniform('gamma', -3, 0),
            'tol': hp.choice('tol', [0.0001]),
            'C': hp.loguniform('C', -3,2)
        }

    if RandomForestRegressor is model_class:
        params = {
                    'n_estimators': scope.int(hp.uniform('n_estimators', 600, 1200)),
                    'max_features': hp.choice('max_features', [1.0,'sqrt']),
                    'max_depth': hp.choice('max_depth', [40, 50, 60]),
                    'min_samples_split': hp.choice('min_samples_split', [5, 7, 9]),
                    'min_samples_leaf': hp.choice('min_samples_leaf', [7, 10, 12]),
                    'criterion': hp.choice('criterion', ['friedman_mse']),
                    'random_state': random_state
                }
    if Ridge is model_class:
        params = {
            'alpha': hp.choice('model__alpha', [1e-15, 1e-10, 1e-8, 1e-3, 1e-2,1,2,5,10,20,25,35, 43,55,100]),
            'random_state': random_state
        }

    if LinearRegression is model_class:
        choices['fit_intercept'] = [True, False]

    if Booster is model_class:
        params = dict(
            # Controls the fraction of features (columns) that will be randomly sampled for each tree.
            colsample_bytree=hp.uniform('colsample_bytree', 0.5, 1.0),
            # Minimum loss reduction required to make a further partition on a leaf node of the tree.
            gamma=hp.uniform('gamma', 0.1, 1.0),
            learning_rate=hp.loguniform('learning_rate', -3, 0),
            # Maximum depth of a tree.
            max_depth=scope.int(hp.quniform('max_depth', 4, 100, 1)),
            min_child_weight=hp.loguniform('min_child_weight', -1, 3),
            # Number of gradient boosted trees. Equivalent to number of boosting rounds.
            # n_estimators=hp.choice('n_estimators', range(100, 1000))
            num_boost_round=hp.quniform('num_boost_round', 500, 1000, 10),
            objective='reg:squarederror',
            # Preferred over seed.
            random_state=random_state,
            # L1 regularization term on weights (xgb’s alpha).
            reg_alpha=hp.loguniform('reg_alpha', -5, -1),
            # L2 regularization term on weights (xgb’s lambda).
            reg_lambda=hp.loguniform('reg_lambda', -6, -1),
            # Fraction of samples to be used for each tree.
            subsample=hp.uniform('subsample', 0.1, 1.0),
        )

    for key, value in choices.items():
        params[key] = hp.choice(key, value)

    if kwargs:
        for key, value in kwargs.items():
            if value is not None:
                kwargs[key] = value

    return params, choices
