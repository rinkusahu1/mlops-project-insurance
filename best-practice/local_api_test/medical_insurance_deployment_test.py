import os
import json
import base64
import pickle
import boto3
import mlflow

import pickle

from flask import Flask, request, jsonify

def get_model_location(run_id):
    model_location = os.getenv('MODEL_LOCATION')

    if model_location is not None:
        return model_location

    model_bucket = os.getenv('MODEL_BUCKET', 'medical-insurance-pp-artifacts')
    experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID', '1')

    model_location = f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/models_mlflow'
    return model_location

def get_scaler_location(run_id):
    scaler_location = os.getenv('SCALER_LOCATION')

    if scaler_location is not None:
        return scaler_location

    model_bucket = os.getenv('MODEL_BUCKET', 'medical-insurance-pp-artifacts')
    experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID', '1')

    scaler_location = f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/preprocessor'

    scaler_path = mlflow.artifacts.download_artifacts(
        artifact_uri=scaler_location,dst_path= "."
    )
    return scaler_path


def load_scaler(run_id):
    scaler_location  = get_scaler_location(run_id)
    with open(f"{scaler_location}/preprocessor.b", "rb") as f_in:
        scaler = pickle.load(f_in)
    return scaler

def load_model(run_id):
    model_path = get_model_location(run_id)
    model = mlflow.pyfunc.load_model(model_path)
    return model


class ModelService:
    def __init__(self, model,scaler, model_version=None, callbacks=None):
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
            'smoker': clean_data['smoker'].get(insurance['smoker'], insurance['smoker']),
            'sex': clean_data['sex'].get(insurance['sex'], insurance['sex']),
            'children': insurance['children'],
            'bmi': insurance['bmi'],
            'age': insurance['age']
        }
        
        return features

    def scaling(self,values):
    # Define the columns to scale
        scalingColumns = ['bmi', 'age']
        for column in scalingColumns:
            if column in values:
                # Use transform on the single value, reshaped to fit transform() requirements
                values[column] = self.stdscaler[column].transform([[values[column]]])[0][0]
        return values

    def prediction_conversion(self,values):
        scalingColumns = ['charges']

        for column in scalingColumns:
            if column in values:
                # Use transform on the single value, reshaped to fit transform() requirements
                values[column] = self.stdscaler[column].inverse_transform([[values[column]]])[0][0]
        print(values)
        return values['charges']

    def predict(self, features):
        features  = [[value for value in features.values()]]
        pred = self.model.predict(features)
        return pred[0]

    def lambda_handler(self, details):
        # print(json.dumps(event))

        
        features = self.prepare_features(details)
        features = self.scaling(features)
        print("prepared featues",features)
        prediction = self.predict(features)
        prediction = self.prediction_conversion({'charges':prediction})
        return {'predictions': prediction}
    

run_id = "a88c9191dd3d471480f1567e10af9d28"
model = load_model(run_id)
scaler  = load_scaler(run_id)

model_service = ModelService(model=model,scaler =scaler, model_version=run_id)

app = Flask('price-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    insurance = request.get_json()

    result = model_service.lambda_handler(insurance)

    return jsonify(result)

# Flask use here, is only use for development use cases, neeed proper wsji server for production env
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)