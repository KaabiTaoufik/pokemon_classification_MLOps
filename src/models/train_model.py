from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import mlflow.keras


# defining model
def cnn(image_size, num_classes):
    classifier = Sequential()
    classifier.add(Conv2D(64, (5, 5), input_shape=image_size, activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(num_classes, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',tf.keras.metrics.AUC(from_logits=True),tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
    return classifier
def train_model(model, training_dataset, validation_dataset, epochs = 3):
    with mlflow.start_run():
        mlflow.keras.autolog()
        filepath = "models/model.h5"
        ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        history = model.fit(training_dataset, validation_data = validation_dataset,callbacks=[ckpt] ,epochs = epochs)
        return model, history
