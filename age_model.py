import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from image_processing import data_generator  # Import the function
import tensorflow as tf

# Directories
train_dir = r"C:\Users\admin\PycharmProjects\Face Recognition\train_data\crop_part1"
valid_dir = r"C:\Users\admin\PycharmProjects\Face Recognition\train_data\UTKFace"
batch_size = 32

# Initialize Data Generators
train_dataset = data_generator(train_dir, batch_size)
val_dataset = data_generator(valid_dir, batch_size)

# Create dataset for age prediction only
train_age_dataset = train_dataset.map(lambda image, labels: (image, labels[0]))
val_age_dataset = val_dataset.map(lambda image, labels: (image, labels[0]))

def create_age_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='linear')  # For regression
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
    return model

# Create and train the age model
age_model = create_age_model()
# Train the model
age_model.fit(
    train_age_dataset,
    validation_data=val_age_dataset,
    epochs=1
)

# Save the model
age_model.save('age_model.h5')
