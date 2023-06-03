import os,random,shutil
from  src.constants import IMG_SIZE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def build_features():
    raw_data_dir = 'data/PokemonData'
    train_dir = 'data/train'
    test_dir = 'data/test'
    file_list = []
    for root,dirs,files in os.walk(raw_data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=42)

    for file in train_files:
        dest_path = file.replace(raw_data_dir, train_dir)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(file, dest_path)

    for file in test_files:
        dest_path = file.replace(raw_data_dir, test_dir)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(file, dest_path)

    target_classes = os.listdir(train_dir)
    datagen=ImageDataGenerator(rescale = 1./255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            )
    training_set=datagen.flow_from_directory(train_dir,
                                            target_size=IMG_SIZE[:2],
                                            batch_size=32,
                                            class_mode='categorical',
                                            color_mode='rgb'
                                            )
    validation_set=datagen.flow_from_directory(test_dir,
                                            target_size=IMG_SIZE[:2],
                                            batch_size=32,
                                            class_mode='categorical',
                                            color_mode='rgb'
                                            )
    return training_set,validation_set, target_classes
