from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from image_processing import data_generator  # Import the function

# Directories
train_dir = 'train_data/UTKFace'
valid_dir = 'train_data/crop_part1'
batch_size = 32

# Initialize Data Generators
train_dataset = data_generator(train_dir, batch_size)
val_dataset = data_generator(valid_dir, batch_size)

# Create dataset for race prediction only
train_race_dataset = train_dataset.map(lambda image, labels: (image, labels[2]))
val_race_dataset = val_dataset.map(lambda image, labels: (image, labels[2]))
def create_race_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5, activation='softmax')  # Multi-class classification
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
race_model = create_race_model()
race_model.fit(train_race_dataset, validation_data = val_race_dataset, epochs=10)
race_model.save('race_model.h5')

