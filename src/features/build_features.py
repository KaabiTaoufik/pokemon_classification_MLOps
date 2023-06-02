import os,random,shutil
from  src.constants import IMG_SIZE
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_features():
    os.makedirs('data/train', exist_ok=True)
    os.system('cp -r data/PokemonData/* data/train/')
    os.makedirs('data/test', exist_ok=True)
    os.system('cp -r data/train/* data/test/')
    os.system("find data/test -name '*.*' -type f -delete")
    train_dir = 'data/train'
    test_dir = 'data/test'

    # Copying 15 random images from train folders to test folders

    for poke in os.listdir(train_dir):
        prep_test_data(poke, train_dir, test_dir)
        
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

def prep_test_data(pokemon, train_dir, test_dir):
    pop = os.listdir(train_dir+'/'+pokemon)
    test_data=random.sample(pop, 15)
    print(test_data)
    for f in test_data:
        shutil.copy(train_dir+'/'+pokemon+'/'+f, test_dir+'/'+pokemon+'/')

if __name__ == '__main__':
    build_features()