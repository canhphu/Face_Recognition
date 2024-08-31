import mtcnn
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load models
age_model = load_model('age_model.h5')
gender_model = load_model('gender_model.h5')
race_model = load_model('race_model.h5')

detector = mtcnn.MTCNN()


def preprocess_face(face_img, target_size=(224, 224)):
    """
    Preprocess face image for model prediction.
    """
    face_img = cv2.resize(face_img, target_size)  # Resize image
    face_img = face_img / 255.0  # Normalize image
    face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
    return face_img


def predict_age_gender_race(face_img):
    """
    Predict age, gender, and race for a given face image.
    """
    face_img = preprocess_face(face_img)

    predicted_age = age_model.predict(face_img)[0][0]
    predicted_gender = gender_model.predict(face_img)[0][0]
    predicted_race = np.argmax(race_model.predict(face_img)[0])

    gender_label = "Male" if predicted_gender > 0.5 else "Female"
    race_labels = ["White", "Black", "Asian", "Indian", "Other"]

    return predicted_age, gender_label, race_labels[predicted_race]


# Initialize Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, width, height = face['box']
        face_img = frame[y:y + height, x:x + width]

        # Predict age, gender, and race
        age, gender, race = predict_age_gender_race(face_img)

        # Draw bounding box and text
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, f'Age: {age:.2f}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f'Gender: {gender}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f'Race: {race}', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show frame from camera
    cv2.imshow('Camera', frame)

    # Push the button 'c' to capture
    key = cv2.waitKey(1)
    if key == ord('c'):
        # Save captured image
        filename = "my_captured_image.jpg"
        cv2.imwrite(filename, frame)
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
