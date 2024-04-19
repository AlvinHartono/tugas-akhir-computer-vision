import cv2
from mtcnn.mtcnn import MTCNN

def detect_faces(frame):
    # Convert frame to RGB (required by MTCNN)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    detector = MTCNN()
    faces = detector.detect_faces(rgb_frame)

    # Draw rectangles around detected faces
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame

# Open video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect faces in the current frame
    frame_with_faces = detect_faces(frame)

    # Display the frame with detected faces
    cv2.imshow('Video', frame_with_faces)

    # Break the loop if any key is pressed
    if cv2.waitKey(1) != -1:
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
