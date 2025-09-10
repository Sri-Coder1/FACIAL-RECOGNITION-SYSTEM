# simple_face_recognition.py
import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_hog_features(image):
    """Extract HOG features from an image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    
    # Initialize HOG descriptor
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # Compute HOG features
    features = hog.compute(gray)
    return features.flatten()

class FaceUtils:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        results = []
        for (x, y, w, h) in faces:
            results.append({'box': (x, y, w, h)})
        return results

    def extract_face(self, frame, box):
        x, y, w, h = box
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        return face_img

    def extract_embeddings(self, image):
        # Placeholder: flatten image as "embedding"
        # Replace with actual model inference for real embeddings
        return image.flatten()[:128]  # Example: first 128 values

def create_directories():
    # Your directory creation logic here
    pass

# The rest of the code would be similar but using extract_hog_features instead of deep learning embeddings