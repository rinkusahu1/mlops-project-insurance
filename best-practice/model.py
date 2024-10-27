import os
import json
import base64
import pickle

import boto3
import mlflow


def get_model_location(run_id):
    model_location = os.getenv('MODEL_LOCATION')

    if model_location is not None:
        return model_location

    model_bucket = os.getenv('MODEL_BUCKET', 'medical-insurance-pp-artifacts')
    experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID', '1')

    model_location = (
        f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/models_mlflow'
    )

    # model_path = mlflow.artifacts.download_artifacts(
    #     artifact_uri=model_location, dst_path="."
    # )
    return model_location


def get_scaler_location(run_id):
    scaler_location = os.getenv('SCALER_LOCATION')

    if scaler_location is not None:
        return f"{scaler_location}/preprocessor.b"

    model_bucket = os.getenv('MODEL_BUCKET', 'medical-insurance-pp-artifacts')
    experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID', '1')

    s3_client = boto3.client('s3')

    file_key = f"{experiment_id}/{run_id}/artifacts/preprocessor/preprocessor.b"
    # scaler_location = (
    #     f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/preprocessor'
    # )

    temp_file_path = '/tmp/preprocessor.b'  # Use the /tmp directory for writing
    s3_client.download_file(model_bucket, file_key, temp_file_path)

    # scaler_path = mlflow.artifacts.download_artifacts(
    #     artifact_uri=scaler_location, dst_path="."
    # )
    return temp_file_path


def load_scaler(run_id):
    scaler_location = get_scaler_location(run_id)
    with open(f"{scaler_location}", "rb") as f_in:
        scaler = pickle.load(f_in)
    return scaler


def load_model(run_id):
    model_path = get_model_location(run_id)
    model = mlflow.pyfunc.load_model(model_path)
    return model


def base64_decode(encoded_data):
    decoded_data = base64.b64decode(encoded_data).decode('utf-8')
    medical_insurance_event = json.loads(decoded_data)
    return medical_insurance_event


class ModelService:
    def __init__(self, model, scaler, model_version=None, callbacks=None):
        self.model = model
        self.model_version = model_version
        self.stdscaler = scaler
        self.callbacks = callbacks or []

    def prepare_features(self, insurance):
        # Define the mapping for clean data
        clean_data = {
            'sex': {'male': 0, 'female': 1},
            'smoker': {'no': 0, 'yes': 1},
        }

        # Apply the mappings to the dictionary values directly
        features = {
            'smoker': clean_data['smoker'].get(
                insurance['smoker'], insurance['smoker']
            ),
            'sex': clean_data['sex'].get(insurance['sex'], insurance['sex']),
            'children': insurance['children'],
            'bmi': insurance['bmi'],
            'age': insurance['age'],
        }

        return features

    def scaling(self, values):
        # Define the columns to scale
        scalingColumns = ['bmi', 'age']
        for column in scalingColumns:
            if column in values:
                # Use transform on the single value, reshaped to fit transform() requirements
                values[column] = self.stdscaler[column].transform([[values[column]]])[
                    0
                ][0]
        return values

    def prediction_conversion(self, values):
        scalingColumns = ['charges']

        for column in scalingColumns:
            if column in values:
                # Use transform on the single value, reshaped to fit transform() requirements
                values[column] = self.stdscaler[column].inverse_transform(
                    [[values[column]]]
                )[0][0]
        print(values)
        return values['charges']

    def predict(self, features):
        features = [list(features.values())]
        pred = self.model.predict(features)
        return pred[0]

    def lambda_handler(self, event):
        # print(json.dumps(event))

        predictions_events = []

        for record in event['Records']:
            encoded_data = record['kinesis']['data']
            medical_insurance_event = base64_decode(encoded_data)

            # print(medical_insurance_event)
            insurance = medical_insurance_event['insurance']
            insurance_id = medical_insurance_event['medical_insurance_id']

            features = self.prepare_features(insurance)
            features = self.scaling(features)
            prediction = self.predict(features)
            prediction = self.prediction_conversion({'charges': prediction})

            prediction_event = {
                'model': 'insurance_price_prediction_model',
                'version': self.model_version,
                'prediction': {
                    'insurance_price': prediction,
                    'insurance_id': insurance_id,
                },
            }

            for callback in self.callbacks:
                callback(prediction_event)

            predictions_events.append(prediction_event)

        return {'predictions': predictions_events}


class KinesisCallback:
    def __init__(self, kinesis_client, prediction_stream_name):
        self.kinesis_client = kinesis_client
        self.prediction_stream_name = prediction_stream_name

    def put_record(self, prediction_event):
        insurance_id = prediction_event['prediction']['insurance_id']

        self.kinesis_client.put_record(
            StreamName=self.prediction_stream_name,
            Data=json.dumps(prediction_event),
            PartitionKey=str(insurance_id),
        )


def create_kinesis_client():
    """
    Creates kinessis client
    """
    endpoint_url = os.getenv('KINESIS_ENDPOINT_URL')

    if endpoint_url is None:
        return boto3.client('kinesis')

    return boto3.client('kinesis', endpoint_url=endpoint_url)


def init(prediction_stream_name: str, run_id: str, test_run: bool):
    model = load_model(run_id)
    scaler = load_scaler(run_id)

    callbacks = []

    if not test_run:
        kinesis_client = create_kinesis_client()
        kinesis_callback = KinesisCallback(kinesis_client, prediction_stream_name)
        callbacks.append(kinesis_callback.put_record)

    model_service = ModelService(
        model=model, scaler=scaler, model_version=run_id, callbacks=callbacks
    )

    return model_service
