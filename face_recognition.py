import mtcnn
TF_ENABLE_ONEDNN_OPTS=0
import matplotlib.pyplot as plt
# load image from file
import cv2

# initialize Camera
cap = cv2.VideoCapture(0)  # Use default camera (0)
# Take photo from camera
while True:
    ret, frame = cap.read()

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

pixels = plt.imread(filename)
print("Shape of image/array:",pixels.shape)
imgplot = plt.imshow(pixels)
plt.show()
detector = mtcnn.MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
for face in faces:
    print(face)


# draw an image with detected objects
def draw_facebox(filename, result_list):
    # load the image
    data = plt.imread(filename)
    # plot the image
    plt.imshow(data)
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = plt.Rectangle((x, y), width, height, fill=False, color='orange')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            dot = plt.Circle(value, radius=20, color='red')
            ax.add_patch(dot)
            # show the plot
        plt.show()
    # show the plot
    plt.show()


# filename = 'test1.jpg' # filename is defined above, otherwise uncomment
# load image from file
# pixels = plt.imread(filename) # defined above, otherwise uncomment
# detector is defined above, otherwise uncomment
# detector = mtcnn.MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image
draw_facebox(filename, faces)

