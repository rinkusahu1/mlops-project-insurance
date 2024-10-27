from typing import Callable, Dict, List, Optional, Tuple, Union

from mlops.utils.data_preparation.feature_engineering import combine_features
from mlops.utils.models.xgboost import build_data
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from xgboost import Booster

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

DEFAULT_INPUTS = [
    {
        # target = "duration": 11.5
        'DOLocationID': 239,
        'PULocationID': 236,
        'trip_distance': 1.98,
    },
    {
        # target = "duration" 20.8666666667
        'DOLocationID': '170',
        'PULocationID': '65',
        'trip_distance': 6.54,
    },
]


@custom
def predict(
    model_settings,
    **kwargs,
) -> List[float]:
    # inputs: List[Dict[str, Union[float, int]]] = kwargs.get('inputs', DEFAULT_INPUTS)
    # inputs = combine_features(inputs)
    print(model_settings)
    return model_settings['sklearn_model_training']
