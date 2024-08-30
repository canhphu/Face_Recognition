import mtcnn
import matplotlib.pyplot as plt
import numpy as np
import cv2

detector = mtcnn.MTCNN()
# detect faces in the image

# initialize Camera
cap = cv2.VideoCapture(0)
# Take photo from camera
while True:
    ret, frame = cap.read()
    faces = detector.detect_faces(frame)
    for face in faces:
        x,y,width,height = face['box']
        face_img = frame[y:y+height, x:x+height]
        face_img = cv2.resize(face_img, (224,224)) #Resize image
        face_img = face_img/255.0
        face_img = np.expand_dims(face_img, axis=0)
    # Show frame from camera
    cv2.imshow('Camera', frame)

    # Push the button 'c' to capture
    key = cv2.waitKey(1)
    if (key == ord('c')):
        # Lưu ảnh vào tệp
        filename = "my_captured_image.jpg"
        cv2.imwrite(filename, frame)
        break
# Release resources
cap.release()
cv2.destroyAllWindows()


