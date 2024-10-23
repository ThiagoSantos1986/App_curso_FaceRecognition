import face_recognition 
import cv2
import os, sys

known_faces_names = []
known_faces_encodings = []

def encoding_faces():
    
    for face in os.listdir('images/samples'):
        img = cv2.imread(f'images/samples/{face}')
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(img_rgb)[0]
        
        known_faces_encodings.append(face_encoding)
        known_faces_names.append(face)
    print("nomes Carregados")
    return known_faces_names, known_faces_encodings

if __name__ == "__main__":
    encoding_faces()
    print(known_faces_names)