import mlflow
import mlflow.keras
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(model: tf.keras.Model, validation_dataset: tf.data.Dataset):
    with mlflow.start_run():
        # Log the model architecture
        mlflow.keras.log_model(model, "model")

        # Evaluate the model on the test set

        loss, accuracy, precision, recall, auc = model.evaluate(validation_dataset)
        print(f"Test loss: {loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test precision: {precision:.4f}")
        print(f"Test recall: {recall:.4f}")
        # print(f"Test f1: {f1}")
        print(f"Test auc: {auc:.4f}")

        # Log the evaluation metrics
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        # mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_auc", auc)

def predict_image(model, pokemon, target_classes):
    img = show_image(pokemon)
    resize = tf.image.resize(img, (64, 64))
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    predicted_pokemon = target_classes[yhat.argmax()]
    print(predicted_pokemon)

def show_image(image):
  img = cv2.imread(image)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img)
  plt.show()
  return img