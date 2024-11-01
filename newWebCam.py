
from imutils.video  import WebcamVideoStream
# from flask import Flask, Request, render_template
import cv2
import face_recognition 
import os, sys, math
import numpy as np
# from model import Pessoa, Encoding
from imutils.video import VideoStream, WebcamVideoStream
from imutils.video import FPS
import imutils
import time
import logging
from logging import DEBUG, basicConfig, info, warning
# from my_connection import DATA_URL


# app = Flask(__name__)

# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['SECRET_KEY'] =  "dev123" #secrets.token_hex()
# # app.config['FLASK_ADMIN_SWATCH'] = 'Cerulean'
# app.config['SQLALCHEMY_DATABASE_URI'] = DATA_URL
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# configure(app)


# @app.route('/')
# def index():
#     fr.run_recogntion()
    
#     return "ola boa noite"



# def filterFrame(frame):
# 	# frame = imutils.resize(frame, width=450)
# 	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	frame = np.dstack([frame, frame, frame])
# 	return frame

def face_distances_percent(face_distances, face_match_media=0.6):
    range = (1.0 - face_match_media) 
    linear_value = (1.0 - face_distances) / (range * 2.0)
    
    if face_distances > face_match_media:
        return str(round(linear_value * 100, 2) + '%')
    else:
        value = (linear_value + ((1.0 -linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


    

class Face_Recognition:
    face_location = []
    face_encoding = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    frame_resizing = 0.25
    
    def __init__(self, src=0):
        self.encoding_faces()
        self.src = src
    
    
    
    def encoding_faces(self):
        
        for face in os.listdir('images/samples'):
            
            # face_load = face_recognition.load_image_file(f'images/samples/{face}')
            img = cv2.imread(f'images/samples/{face}')
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(img_rgb)[0]
            
            # unpack_b5 = np.frombuffer(face_encoding, dtype=np.float64)
            # self.known_face_encodings.append(unpack_b5)
                
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(face)
            
        # print(self.known_face_names)         
      
    def detect_faces(self, frame):
        
        frame_small = cv2.resize(frame, (0,0), fx=self.frame_resizing, fy=self.frame_resizing)
              
        self.face_location = face_recognition.face_locations(frame_small, number_of_times_to_upsample=1)           
        self.face_encodings = face_recognition.face_encodings(frame_small, self.face_location)
                
        self.faces_names = []
        
        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                        
            name = "Unknown"
                   
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]            
               
                info(f'Rosto Localizado {name}')
                        
            else:
                warning(f"{name} in local..")
                    
            self.faces_names.append(name)        
                
                    
        # self.face_location = np.array(self.face_location)
        # self.face_location = self.face_location / self.frame_resizing
        return self.face_location , self.faces_names
        
            
    def gen_frames(self ):
        
        process_current_frame = True
            
        # camera = WebcamVideoStream(src=self.src).start()
        camera = VideoStream(src=self.src).start()
            
        time.sleep(1.0)    
        fps = FPS().start()
            
        while True:
            
            frame = camera.read()
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.dstack([frame, frame, frame])
            # frame = self.filterFrame(frame)
           
                
            if process_current_frame:
                
                self.face_locations, self.faces_names = self.detect_faces(frame)

            self.process_current_frame = not self.process_current_frame  
            
            if self.face_locations:    
                
                for (top, right, bottom, left), name in zip(self.face_locations, self.faces_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                        
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left, bottom ), font, 1.0, (255, 255, 255), 1)
                        
            cv2.imshow('Video ', frame)
            
            fps.update()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
              
        camera.stop()
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    
    basicConfig(level=DEBUG,
                        filename='TESTANDO.log',
                        filemode='w',
                        format='%(levelname)s: %(name)s : %(asctime)s: %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S'
            )
    rtmp_url = "rtmp://192.168.0.25:1935/webcam/teste"

    ip = 'rtsp://192.168.0.35:5555/h264.sdp'
    print("iniciando a captura de imagens... ")
    fr = Face_Recognition(src=ip)
    fr.gen_frames()
   
