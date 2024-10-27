if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from typing import Callable, Dict, List, Optional, Tuple, Union

from mlops.utils.data_preparation.feature_engineering import combine_features
from mlops.utils.models.xgboost import build_data
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from xgboost import Booster


@transformer
def predict(
    model_settings,
    **kwargs,
) -> List[float]:

    print(model_settings)
    model, vectorizer = model_settings

    return "Hello"
