from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
import mlflow


# defining model
def cnn(image_size, num_classes):
    classifier = Sequential()
    classifier.add(Conv2D(64, (5, 5), input_shape=image_size, activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(num_classes, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return classifier
def train_model(model, training_dataset, validation_dataset, epochs = 2):
    with mlflow.start_run():
        mlflow.tensorflow.log_model(model, "model")
        mlflow.tensorflow.autolog()
        history = model.fit(training_dataset, validation_data = validation_dataset, epochs = epochs)
        mlflow.tensorflow.log_model(model, "model")
        return model, history
