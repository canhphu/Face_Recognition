import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from image_processing import create_data_generator  # Import the function

# Directories
root_directory = 'train_data'
train_dir = os.path.join(root_directory, 'UTKFace')
val_dir = os.path.join(root_directory, 'crop_part1')

def create_age_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')  # Example for regression
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
    return model

# Create data generators
train_generator = create_data_generator(train_dir, target_size=(200, 200), class_mode=None)
val_generator = create_data_generator(val_dir, target_size=(200, 200), class_mode=None)

# Create and train the age model
age_model = create_age_model()
age_model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save the model
age_model.save('age_model.h5')
