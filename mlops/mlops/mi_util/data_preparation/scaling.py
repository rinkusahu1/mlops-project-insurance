from typing import Dict, List, Optional, Tuple

import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler

def scaling(df,stdscaler):
    scalingColumns = ['bmi','age','charges']
    for column in scalingColumns:
        if column == 'charges':
            df[column] = stdscaler[column].transform(df[[column]])  # Fit the scaler to the column
        else:
            df[column] = stdscaler[column].transform(df[[column]])
    return df

def scale_features(
    training_set: pd.DataFrame,
    validation_set: Optional[pd.DataFrame] = None,
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, Dict]:
    
    scalers = {}

    # Iterate over each column and create a scaler for it
    for column in ['bmi','age','charges']:
        scaler = StandardScaler()
        if column == 'charges':
            scaler.fit(training_set[[column]])  # Fit the scaler to the column
        else:
            scaler.fit(training_set[[column]])
        scalers[column] = scaler  # Store the scaler in the dictionary
    
    X_train  = scaling(training_set,scalers)
    X_train['bmi'] = X_train.bmi.values.reshape(-1,1)
    X_train['age'] = X_train.age.values.reshape(-1,1)
    X_train['charges'] = X_train.charges.values.reshape(-1,1)

    X_val = None
    
    if validation_set is not None:
        X_val = scaling(validation_set,scalers)
        X_val['bmi'] = X_val.bmi.values.reshape(-1,1)
        X_val['age'] = X_val.age.values.reshape(-1,1)
        X_val['charges'] = X_val.charges.values.reshape(-1,1)


    return X_train, X_val, scalers
