from typing import List, Tuple, Dict

from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.mi_util.data_preparation.scaling import scale_features

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_exporter
def export(
    data: Tuple[DataFrame, DataFrame, DataFrame], *args, **kwargs
) -> Tuple[
    csr_matrix,
    csr_matrix,
    csr_matrix,
    Series,
    Series,
    Series,
    Dict,
]:
    df, df_train, df_val = data
    target = kwargs.get('target', 'charges')

    X, _, _ = scale_features(df)
    y: Series = df[target]

    X_train, X_val, dv = scale_features(
        df_train,
        df_val,
    )
    X_train = X_train.drop(columns=[target])
    X_val = X_val.drop(columns=[target])

    y_train = df_train[target]
    y_val = df_val[target]

    return X, X_train, X_val, y, y_train, y_val, dv