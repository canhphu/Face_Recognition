from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from image_processing import data_generator  # Import the function

# Directories
train_dir = 'train_data/UTKFace'
valid_dir = 'train_data/crop_part1'
batch_size = 32

# Initialize Data Generators
train_dataset = data_generator(train_dir, batch_size)
val_dataset = data_generator(valid_dir, batch_size)

# Create dataset for gender prediction only
train_gender_dataset = train_dataset.map(lambda image, labels: (image, labels[1]))
val_gender_dataset = val_dataset.map(lambda image, labels: (image, labels[1]))

def create_gender_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train the age model
gender_model = create_gender_model()
# Train the model
gender_model.fit(
    train_gender_dataset,
    validation_data=val_gender_dataset,
    epochs=1
)
gender_model.save('gender_model.h5')

