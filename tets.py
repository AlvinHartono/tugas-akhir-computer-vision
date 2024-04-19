from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image
import cv2

def extract_faces_from_image(image_path, required_size=(224, 224)):
    # load image and detect faces
    image = plt.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    for face in faces:
        # extract the bounding box from the requested face
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face_boundary = image[y1:y2, x1:x2]

        # resize pixels to the model size
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

    return face_images, faces

# Load the image
image_path = '4org.jpg'
image = cv2.imread(image_path)

# Detect faces and extract faces
detector = MTCNN()
extracted_faces, faces_info = extract_faces_from_image(image_path)
plt.figure(figsize=(8, 4))
for i, face in enumerate(extracted_faces):
    plt.subplot(1, len(extracted_faces), i + 1)
    plt.imshow(face)
    plt.axis('off')
plt.show()

# Print detected faces information
print(faces_info)
