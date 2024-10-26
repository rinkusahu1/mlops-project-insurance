from mlflow.tracking import MlflowClient
import mlflow
import pickle

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("medical-insurance-price_prediction")
client = MlflowClient("http://127.0.0.1:5000")


def find_experiment_id(name, experiments):
    for exp in experiments:
        if exp.name == name:
            return exp.experiment_id
    return None

def latest_registered_model_details(registere_model_name,experiment_id):
    registered_prediction_model = mlflow.search_registered_models(filter_string = f"name='{registere_model_name}'")
    
    if len(registered_prediction_model[0].latest_versions) != 0:
        registered_run_id = registered_prediction_model[0].latest_versions[0].run_id
        registered_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"attributes.run_id  ='{registered_run_id}'",
            max_results=1  # Adjust max_results as needed
        )
        rmse = None
        if registered_runs[0].data.metrics.get('val_rmse') is not None:
            rmse = registered_runs[0].data.metrics.get('val_rmse')
        else:
            rmse = registered_runs[0].data.metrics.get('rmse')
        return registered_runs[0].info.run_id, rmse
    else:
        return None,None


def register_model(
    exp_name = 'medical-insurance-price_prediction',
    model_class = 'RandomForestRegressor',
    registered_model_name = 'MedicalInsuranceChargesPrediction'):

    experiment_id = find_experiment_id(exp_name, mlflow.search_experiments())

    reg_run_id,reg_rmse = latest_registered_model_details(registered_model_name,experiment_id)

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.model LIKE '%{model_class}%'",
        order_by=["metrics.rmse ASC"],  # Sort by RMSE in ascending order
        max_results=5  # Adjust max_results as needed
    )

    if reg_rmse == None  or reg_rmse > runs[0].data.metrics['val_rmse']:
        run_id = runs[0].info.run_id
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/models",
            name=registered_model_name,
            tags ={"Stage":"Staging"},
        )

        if reg_rmse == None:
            print(f"Registered First version of model with rmse:{runs[0].data.metrics['val_rmse']}")
        else:
            print(f"Latest run id's {runs[0].info.run_id} performance improved by {reg_rmse - runs[0].data.metrics['val_rmse']} rmse. old rmse {reg_rmse}")
        return True
    else :
        print(f"Model Performance doesn't improve, already registered run id's {reg_run_id}, rmse:{reg_rmse} performance is good till now.")
        return False

def load_registered_model( 
    exp_name = 'medical-insurance-price_prediction',
    registered_model_name = 'MedicalInsuranceChargesPrediction'):

    experiment_id = find_experiment_id(exp_name, mlflow.search_experiments())

    reg_run_id,reg_rmse = latest_registered_model_details(registered_model_name,experiment_id)

    # logged_model = f'runs:/{reg_run_id}/models_mlflow'

    # loaded_model = mlflow.pyfunc.load_model(logged_model)

# Load model as a PyFuncModel.
    client.download_artifacts(run_id=reg_run_id, path='models_mlflow', dst_path='.')
    client.download_artifacts(run_id=reg_run_id, path='preprocessor', dst_path='.')

    with open("preprocessor/preprocessor.b", "rb") as f_in:
        dv = pickle.load(f_in)
     
    loaded_model = mlflow.pyfunc.load_model('models_mlflow')
    return loaded_model, dv
