from typing import List, Tuple, Union

from pandas import DataFrame, Index
from sklearn.model_selection import train_test_split


def split(
    df: DataFrame, return_indexes: bool = False, seed: int = 42
) -> Union[Tuple[DataFrame, DataFrame], Tuple[Index, Index]]:

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed)
    if return_indexes:
        return train_df.index, val_df.index

    return train_df, val_df
