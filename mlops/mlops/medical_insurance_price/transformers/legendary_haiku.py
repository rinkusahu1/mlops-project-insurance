if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from typing import Dict, List, Tuple, Union

from sklearn.feature_extraction import DictVectorizer
from xgboost import Booster

from mlops.utils.data_preparation.feature_engineering import combine_features
from mlops.utils.models.xgboost import build_data
from typing import Callable, Dict, Optional, Tuple, Union
from sklearn.base import BaseEstimator

@transformer
def predict(
    model_settings,
    **kwargs,
) -> List[float]:

    print(model_settings)
    model, vectorizer = model_settings

    return "Hello"

