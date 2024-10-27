if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from io import BytesIO, StringIO

import pandas as pd
import requests


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    df = pd.read_csv(
        'https://raw.githubusercontent.com/rinkusahu1/mlops-project-insurance/refs/heads/model_building/insurance.csv'
    )
    dfs.append(df)

    return pd.concat(dfs)
