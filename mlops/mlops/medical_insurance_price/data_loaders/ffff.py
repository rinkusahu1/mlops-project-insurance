if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd
from mlops.mi_util.analyticsm.data import load_data


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    df = load_data()

    return df
