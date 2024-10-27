from typing import Callable, Dict, Tuple, Union

from mlops.mi_util.model_registration.register import register_model
from mlops.mi_util.models.sklearn import (
    load_class,
    train_model,
    train_model_with_logging,
)
from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def train(
    settings: Tuple[
        Dict[str, Union[bool, float, int, str]],
        csr_matrix,
        csr_matrix,
        Series,
        Series,
        Dict,
        Dict[str, Union[Callable[..., BaseEstimator], str]],
    ],
    **kwargs,
) -> Tuple[BaseEstimator, Dict]:
    hyperparameters, X_train, X_val, y_train, y_val, dv, model_info = settings

    model_class = model_info['cls']
    model, metrics = train_model_with_logging(
        model_class, hyperparameters, X_train, y_train, X_val, y_val, dv
    )

    if register_model():
        print("New version of model registered")
    else:
        print("Performance didn't improve")

    return model, dv
