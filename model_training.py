"""Train an improved classifier on extracted features"""    
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

def train_model():
    # Load features and labels
    with open('models/embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
    
    embeddings = np.array(data['embeddings'])
    labels = data['labels']
    
    # Check if we have enough data
    if len(np.unique(labels)) < 2:
        print("Error: Need at least 2 different people for classification")
        return
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    # Create pipeline with scaling and classifier
    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'  # Handles imbalanced classes
        )
    )
    
    # Train classifier
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Model accuracy: {accuracy:.2f}%")
    
    # Cross-validation for better accuracy estimate
    cv_scores = cross_val_score(pipeline, embeddings, encoded_labels, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
    
    # Save the trained model and label encoder
    with open('models/face_recognizer.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("Model training completed!")

if __name__ == "__main__":
    train_model()