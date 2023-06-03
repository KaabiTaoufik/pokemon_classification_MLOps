import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, Recall, AUC
import mlflow

def evaluate_model(model: tf.keras.Model, validation_dataset: tf.data.Dataset):
        with mlflow.start_run():
            loss, accuracy ,  auc , recall  , precision = model.evaluate(validation_dataset)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("loss", loss)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("auc", auc)