import mlflow

def setup_mlflow(experiment_name):
    mlflow.set_tracking_uri("http://mlops.uksouth.cloudapp.azure.com:5000")
    mlflow.set_experiment(experiment_name)