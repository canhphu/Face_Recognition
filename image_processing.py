from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

root_directory = 'train_data'

val_dir = os.path.join(root_directory, 'crop_part1')
train_dir = os.path.join(root_directory, 'UTKFace')

total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(val_dir)])
def create_data_generator(data_dir, target_size=(200, 200), batch_size=32, class_mode=None):
    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(
        directory=data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=True
    )

    return generator

train_generator = create_data_generator(train_dir, target_size=(200, 200), class_mode=None)
val_generator = create_data_generator(val_dir, target_size=(200,200), class_mode=None)