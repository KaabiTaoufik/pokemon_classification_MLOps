import mlflow
import mlflow.keras
import tensorflow as tf

def initiate_mlflow_experiment(experiment_name):
    mlflow.set_experiment(experiment_name)
