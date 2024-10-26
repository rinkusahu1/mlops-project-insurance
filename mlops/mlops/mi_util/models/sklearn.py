from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import sklearn
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from mlops.mi_util.hyperparameters.shared import build_hyperparameters_space, derived_best_hyperparamteres
import mlflow
import pickle

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("medical-insurance-price_prediction")

HYPERPARAMETERS_WITH_CHOICE_INDEX = [
    'fit_intercept',
]


def load_class(module_and_class_name: str) -> BaseEstimator:
    """
    module_and_class_name:
        ensemble.ExtraTreesRegressor
        ensemble.GradientBoostingRegressor
        ensemble.RandomForestRegressor
        linear_model.Lasso
        linear_model.LinearRegression
        svm.LinearSVR
    """
    parts = module_and_class_name.split('.')
    cls = sklearn
    for part in parts:
        cls = getattr(cls, part)

    return cls


def train_model(
    model: BaseEstimator,
    X_train: csr_matrix,
    y_train: Series,
    X_val: Optional[csr_matrix] = None,
    eval_metric: Callable = mean_squared_error,
    fit_params: Optional[Dict] = None,
    y_val: Optional[Series] = None,
    **kwargs,
) -> Tuple[BaseEstimator, Optional[Dict], Optional[np.ndarray]]:
    
    model.fit(X_train, y_train, **(fit_params or {}))

    metrics = None
    y_pred = None
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        y_train_pred =  model.predict(X_train)

        r2_train = r2_score(y_train, y_train_pred)
        r2_val = r2_score(y_val, y_pred)
        train_rmse = eval_metric(y_train, y_train_pred, squared=False)
        train_mse = eval_metric(y_train, y_train_pred, squared=True)
        val_rmse = eval_metric(y_val, y_pred, squared=False)
        val_mse = eval_metric(y_val, y_pred, squared=True)
        metrics = dict(val_mse=val_mse, val_rmse=val_rmse,train_mse =train_mse,train_rmse =train_rmse,r2_train=r2_train,r2_val=r2_val)

    return model, metrics, y_pred

def train_model_with_logging(
    model_class: Callable[..., BaseEstimator],
    params: Dict,
    X_train: csr_matrix,
    y_train: Series,
    X_val: csr_matrix,
    y_val: Series,
    dv: Dict
) ->  Tuple[BaseEstimator, Optional[Dict]]:
    with mlflow.start_run():
        mlflow.set_tag("developer", "rinku")
        mlflow.set_tag("model",model_class)
        mlflow.set_tag("run_type","model")
        print("Model param",params)
        mlflow.log_params(params)
        
        model, metrics, predictions = train_model(
            model_class(**params),
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val
        )
        mlflow.log_metrics(metrics)
        
        with open("preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor.b", artifact_path="preprocessor")
        mlflow.sklearn.log_model(model, artifact_path="models_mlflow")

        return model, metrics

def tune_hyperparameters(
    model_class: Callable[..., BaseEstimator],
    X_train: csr_matrix,
    y_train: Series,
    X_val: csr_matrix,
    y_val: Series,
    callback: Optional[Callable[..., None]] = None,
    eval_metric: Callable[[Series, Series], float] = mean_squared_error,
    fit_params: Optional[Dict] = None,
    hyperparameters: Optional[Dict] = None,
    max_evaluations: int = 50,
    random_state: int = 42,
) -> Dict:
    def __objective(
        params: Dict,
        X_train=X_train,
        X_val=X_val,
        callback=callback,
        eval_metric=eval_metric,
        fit_params=fit_params,
        model_class=model_class,
        y_train=y_train,
        y_val=y_val,
    ) -> Dict[str, Union[float, str]]:
        with mlflow.start_run():
            mlflow.set_tag("developer", "rinku")
            mlflow.set_tag("model",model_class)
            mlflow.set_tag("run_type","hp_tuning")
            print("Model param",params)
            mlflow.log_params(params)
            if fit_params is not None:
                mlflow.log_params(fit_params)
            model, metrics, predictions = train_model(
                model_class(**params),
                X_train,
                y_train,
                X_val=X_val,
                y_val=y_val,
                eval_metric=eval_metric,
                fit_params=fit_params,
            )
            mlflow.log_metrics(metrics)
            if callback:
                callback(
                    hyperparameters=params,
                    metrics=metrics,
                    model=model,
                    predictions=predictions,
                )

            return dict(loss=metrics['val_rmse'], status=STATUS_OK)

    space, choices = build_hyperparameters_space(
        model_class,
        random_state=random_state,
        **(hyperparameters or {}),
    )

    best_hyperparameters = fmin(
        fn=__objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evaluations,
        trials=Trials(),
    )

    # Convert choice index to choice value.
    for key in HYPERPARAMETERS_WITH_CHOICE_INDEX:
        if key in best_hyperparameters and key in choices:
            idx = int(best_hyperparameters[key])
            best_hyperparameters[key] = choices[key][idx]

    # fmin will return max_depth as a float for some reason
    for key in [
        'max_depth',
        'max_iter',
        'min_samples_leaf',
        'min_samples_split',
        'n_estimators',
    ]:
        if key in best_hyperparameters:
            best_hyperparameters[key] = int(best_hyperparameters[key])
    
    best_hyperparameters = derived_best_hyperparamteres(model_class,best_hyperparameters)

    return best_hyperparameters
