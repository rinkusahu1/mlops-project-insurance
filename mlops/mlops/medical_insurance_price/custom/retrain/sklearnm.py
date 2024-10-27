from mage_ai.orchestration.triggers.api import trigger_pipeline

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def retrain(*args, **kwargs):
    models = [
        "linear_model.LinearRegression",
        "linear_model.Ridge",
        "svm.SVR",
        "ensemble.RandomForestRegressor",
    ]

    trigger_pipeline(
        'sklearn_training_m',
        check_status=True,
        error_on_failure=True,
        schedule_name='Automatic retraining for sklearn models',
        verbose=True,
    )
