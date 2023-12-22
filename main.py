# Import necessary libraries
import cv2
import numpy as np

# Load the pre-trained face detection model from OpenCV
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_classifier = cv2.CascadeClassifier('haarcascade_smile.xml')

# Define a function to detect faces in an image
import cv2

# Load the face and smile classifiers
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_classifier = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect_faces_and_smiles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Extract the face ROI (Region of Interest)
        face_roi = gray[y:y+h, x:x+w]

        # Detect smiles within the face ROI
        smiles = smile_classifier.detectMultiScale(face_roi, 1.8, 20)
        
        for (sx, sy, sw, sh) in smiles:
            # Draw a rectangle around each smile
            cv2.rectangle(img, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 255, 0), 2)

    return img

# Webcam capture code remains the same
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = detect_faces_and_smiles(frame)
    cv2.imshow('Face and Smile Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
