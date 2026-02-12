import cv2
import face_recognition
import os
import numpy as np

class FaceEngine:
    def __init__(self, dataset_path='dataset/'):
        self.dataset_path = dataset_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_registered_faces()

    def load_registered_faces(self):
        """Loads images from dataset folder and encodes them."""
        for filename in os.listdir(self.dataset_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                img = face_recognition.load_image_file(f"{self.dataset_path}{filename}")
                # Get the 128-d encoding
                encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(encoding)
                # Use filename as the person's name
                self.known_face_names.append(os.path.splitext(filename)[0])

    def recognize_face(self, frame):
        """Identifies faces in a live frame."""
        # Convert BGR (OpenCV) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            # Use the shortest distance for better accuracy
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            
            face_names.append(name)
        return face_locations, face_names