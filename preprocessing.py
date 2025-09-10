"""Preprocess all images in the known_faces directory"""
import cv2
import os
import numpy as np
from utils import FaceUtils, create_directories

def preprocess_dataset():

    create_directories()
    utils = FaceUtils()
    
    for person_name in os.listdir('known_faces'):
        person_dir = os.path.join('known_faces', person_name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"Preprocessing images for {person_name}")
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path)
            
            if image is not None:
                # Detect and align face
                faces = utils.detect_faces(image)
                if faces:
                    # Use the first face found
                    face_box = faces[0]['box']
                    aligned_face = utils.extract_face(image, face_box)
                    
                    # Save the preprocessed image
                    cv2.imwrite(image_path, aligned_face)
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    preprocess_dataset()