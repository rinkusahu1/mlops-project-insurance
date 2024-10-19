from typing import Tuple

import pandas as pd

from mlops.mi_util.data_preparation.categorical_encoder import categorical_encoder
from mlops.mi_util.data_preparation.feature_selector import select_features
from mlops.mi_util.data_preparation.splitters import split

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = kwargs.get('seed')

    df = select_features(df)
    df = categorical_encoder(df)

    df_train, df_val = split(
        df,
        seed = seed
    )

    return df, df_train, df_val