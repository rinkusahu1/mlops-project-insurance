import pandas as pd

CATEGORICAL_FEATURES = ['smoker','sex','children']

def categorical_encoder(df: pd.DataFrame) -> pd.DataFrame:
    clean_data = {'sex': {'male' : 0 , 'female' : 1} ,
                 'smoker': {'no': 0 , 'yes' : 1},
               }
    df.replace(clean_data, inplace=True)
    return df