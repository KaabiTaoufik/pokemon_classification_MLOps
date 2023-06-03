import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, Recall, AUC
import mlflow

def evaluate_model(model: tf.keras.Model, validation_dataset: tf.data.Dataset):
        accuracy_metric = Accuracy()
        recall_metric = Recall()
        auc_metric = AUC()
        for images, labels in validation_dataset:
            predictions = model.predict(images)
            accuracy_metric.update_state(labels, predictions)
            recall_metric.update_state(labels, predictions)
            auc_metric.update_state(labels, predictions)
        accuracy = accuracy_metric.result().numpy()
        recall = recall_metric.result().numpy()
        auc = auc_metric.result().numpy()
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("auc", auc)