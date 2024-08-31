import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def parse_filename(filename):
    """
    Extract labels from the filename formatted as: {age}_{gender}_{race}_{date}.jpg
    """
    age, gender, race, _ = filename.split('_')
    return int(age), int(gender), int(race)


def load_data_from_directory(data_dir, target_size=(224, 224)):
    """
    Load data from a specific directory and extract labels from the filename.
    """
    images = []
    ages = []
    genders = []
    races = []

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith(".jpg"):
                img_path = os.path.join(root, filename)
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img)
                images.append(img_array)

                age, gender, race = parse_filename(filename)
                ages.append(age)
                genders.append(gender)
                races.append(race)

    # Convert lists to numpy arrays
    images = np.array(images)
    ages = np.array(ages)
    genders = np.array(genders)
    races = np.array(races)

    # Normalize images
    images = images / 255.0

    return images, ages, genders, races


def load_and_preprocess_data(train_dir, val_dir, target_size=(224, 224)):
    """
    Load and preprocess data from separate train and validation directories.
    """
    train_images, train_ages, train_genders, train_races = load_data_from_directory(train_dir, target_size)
    val_images, val_ages, val_genders, val_races = load_data_from_directory(val_dir, target_size)

    return (train_images, train_ages, train_genders, train_races), \
        (val_images, val_ages, val_genders, val_races)
