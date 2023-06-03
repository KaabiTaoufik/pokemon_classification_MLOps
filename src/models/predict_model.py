import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np



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
