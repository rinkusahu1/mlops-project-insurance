from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import model


def read_text(file):
    test_directory = Path(__file__).parent

    with open(test_directory / file, 'rt', encoding='utf-8') as f_in:
        return f_in.read().strip()


def prepare_scaler():
    data = {
        'bmi': np.array([22.5, 24.0, 30.5, 28.7, 26.1]).reshape(-1, 1),
        'age': np.array([25, 30, 45, 35, 40]).reshape(-1, 1),
        'charges': np.array([5000, 6000, 12000, 9000, 15000]).reshape(-1, 1),
    }

    # Create a dictionary to hold the scalers
    scalers = {
        'bmi': StandardScaler(),
        'age': StandardScaler(),
        'charges': StandardScaler(),
    }

    # Fit the scalers to the respective data
    scalers['bmi'].fit(data['bmi'])
    scalers['age'].fit(data['age'])
    scalers['charges'].fit(data['charges'])

    return scalers


def test_base64_decode():
    base64_input = read_text('data.b64')

    actual_result = model.base64_decode(base64_input)
    expected_result = {
        "insurance": {
            "smoker": "yes",
            "sex": "female",
            "children": 0,
            "bmi": 26.29,
            "age": 62,
        },
        "medical_insurance_id": 256,
    }

    assert actual_result == expected_result


def test_prepare_features():
    model_service = model.ModelService(None, None)
    insurance = {
        "smoker": "yes",
        "sex": "female",
        "children": 0,
        "bmi": 26.29,
        "age": 62,
    }

    actual_features = model_service.prepare_features(insurance)

    expected_fetures = {"smoker": 1, "sex": 1, "children": 0, "bmi": 26.29, "age": 62}

    assert actual_features == expected_fetures


class ModelMock:
    def __init__(self, value):
        self.value = value

    def scaling(self, values):
        return {"smoker": 1, "sex": 1, "children": 0, "bmi": 26.29, "age": 62}

    def predict(self, X):
        n = len(X)
        return [self.value] * n


def test_predict():
    model_mock = ModelMock(10.0)
    model_service = model.ModelService(model_mock, scaler={})

    features = {"smoker": 1, "sex": 1, "children": 0, "bmi": 26.29, "age": 62}

    actual_prediction = model_service.predict(features)
    expected_prediction = 10.0

    assert actual_prediction == expected_prediction


def test_lambda_handler():
    model_mock = ModelMock(10.0)
    model_version = 'Test123'
    scaler = prepare_scaler()
    model_service = model.ModelService(
        model=model_mock, scaler=scaler, model_version=model_version
    )

    base64_input = read_text('data.b64')

    event = {
        "Records": [
            {
                "kinesis": {
                    "data": base64_input,
                },
            }
        ]
    }

    actual_predictions = model_service.lambda_handler(event)
    expected_predictions = {
        'predictions': [
            {
                'model': 'insurance_price_prediction_model',
                'version': model_version,
                'prediction': {
                    'insurance_price': 46602.15047547655,
                    'insurance_id': 256,
                },
            }
        ]
    }

    assert actual_predictions == expected_predictions
