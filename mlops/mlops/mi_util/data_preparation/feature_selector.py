from typing import List, Optional

import pandas as pd

CATEGORICAL_FEATURES = ['smoker', 'sex', 'children']
NUMERICAL_FEATURES = ['bmi', 'age', 'charges']


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    return df[columns]
