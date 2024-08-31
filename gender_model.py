from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from image_processing import load_and_preprocess_data


def build_gender_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification for gender
    ])
    return model


def train_gender_model(train_dir, val_dir):
    (train_images, _, train_genders, _), (val_images, _, val_genders, _) = load_and_preprocess_data(train_dir, val_dir)

    model = build_gender_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_images, train_genders, validation_data=(val_images, val_genders), epochs=10)
    model.save('gender_model.h5')


if __name__ == '__main__':
    train_gender_model('train_data/UTKFace', 'train_data/crop_part1')
