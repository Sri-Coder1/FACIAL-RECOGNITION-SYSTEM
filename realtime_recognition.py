"""Perform real-time face recognition"""
import cv2
import pickle
import numpy as np
from utils import FaceUtils

def realtime_recognition():

    utils = FaceUtils()
    
    # Load trained model and label encoder
    try:
        with open('models/face_recognizer.pkl', 'rb') as f:
            classifier = pickle.load(f)
        
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    except:
        print("No trained model found. Please train the model first.")
        return
    
    cap = cv2.VideoCapture(0)
    
    print("Starting real-time face recognition. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        faces = utils.detect_faces(frame)
        
        for face in faces:
            x, y, w, h = face['box']
            face_img = utils.extract_face(frame, (x, y, w, h))

            # Get embedding and predict
            embedding = utils.extract_embeddings(face_img)
            prediction = classifier.predict([embedding])
            probability = classifier.predict_proba([embedding]).max()

            # Get label name or show "unknown face"
            if probability > 0.7:
                label = label_encoder.inverse_transform(prediction)[0]
                color = (0, 255, 0)  # Green for known face
            else:
                label = "unknown face"
                color = (0, 0, 255)  # Red for unknown face

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    realtime_recognition()