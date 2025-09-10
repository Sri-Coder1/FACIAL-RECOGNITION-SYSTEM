''' Collecting face data using webcam and save to known_faces directory '''
import cv2
import os
from utils import FaceUtils, create_directories

def collect_face_data(person_name, num_samples=30):
    """
    Collect face data using webcam
    """
    create_directories()
    utils = FaceUtils()
    cap = cv2.VideoCapture(0)
    count = 0
    person_dir = os.path.join('known_faces', person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    print(f"Collecting {num_samples} samples for {person_name}. Press 'q' to quit.")
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
            
        faces = utils.detect_faces(frame)
        
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)
            face_img = frame[y:y+h, x:x+w]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Sample: {count}/{num_samples} | Press 'r' to capture", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Wait for 'r' key to capture
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') and count < num_samples:
                cv2.imwrite(os.path.join(person_dir, f"{person_name}_{count+1}.jpg"), face_img)
                count += 1
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print(f"Collected samples for {person_name}")
                return

            if count >= num_samples:
                break
        
        cv2.imshow('Collecting Face Data', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == "__main__":
    person_name = input("Enter the person's name: ")
    collect_face_data(person_name, num_samples=30)
    cv2.destroyAllWindows()
    print(f"Collected samples for {person_name}")
