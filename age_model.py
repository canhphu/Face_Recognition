# build_age_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from image_processing import load_and_preprocess_data


def build_age_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')  # Regression for age prediction
    ])
    return model


def train_age_model(train_dir, val_dir):
    (train_images, train_ages, _, _), (val_images, val_ages, _, _) = load_and_preprocess_data(train_dir, val_dir)

    model = build_age_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(train_images, train_ages, validation_data=(val_images, val_ages), epochs=10)
    model.save('age_model.h5')


if __name__ == '__main__':
    train_age_model('path/to/train_utkface', 'path/to/val_utkface')
