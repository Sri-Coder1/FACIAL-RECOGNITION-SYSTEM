"""Extract features from all preprocessed images"""
import cv2
import os
import numpy as np
import pickle
from utils import FaceUtils, create_directories

def extract_features():
   
    create_directories()
    utils = FaceUtils()
    
    embeddings = []
    labels = []
    
    for person_name in os.listdir('known_faces'):
        person_dir = os.path.join('known_faces', person_name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"Extracting features for {person_name}")
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path)
            
            if image is not None:
                embedding = utils.extract_embeddings(image)
                embeddings.append(embedding)
                labels.append(person_name)
    
    # Save features and labels
    os.makedirs('models', exist_ok=True)
    with open('models/embeddings.pkl', 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'labels': labels}, f)
    
    print(f"Feature extraction completed! Saved {len(embeddings)} samples.")

if __name__ == "__main__":
    extract_features()