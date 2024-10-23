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
import asyncio
from load_names_encoding import encoding_faces

class Faces_Recognition:
    face_location = []
    face_encoding = []
    face_names = []
    known_faces_encodings = []
    known_faces_names = []
    process_current_frame = True
    frame_resizing = 0.25
    
    def __init__(self, src=0):
        self.encoding_faces()
        self.src = src
    
    def encoding_faces(self,):
    
        for face in os.listdir('images/samples'):
            img = cv2.imread(f'images/samples/{face}')
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(img_rgb)[0]
            
            self.known_faces_encodings.append(face_encoding)
            self.known_faces_names.append(face)
        print(self.known_faces_names)




    def detect_faces(self, frame):
                    
            self.face_location = face_recognition.face_locations(frame, number_of_times_to_upsample=1)
            yield (self.face_location)           
            
            self.face_encodings = face_recognition.face_encodings(frame, self.face_location)
                    
            self.faces_names = []
            
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_faces_encodings, face_encoding, tolerance=0.6)
                            
                name = "Unknown"
                    
                face_distances = face_recognition.face_distance(self.known_faces_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_faces_names[best_match_index]            
                
                        
                self.faces_names.append(name)        
             
            return  self.face_locations, self.faces_names

    def gen_frames(self ):
            
            process_current_frame = True
                
            camera = WebcamVideoStream(self.src).start()
            # camera = VideoStream(src=ip).start()
                
            time.sleep(1.0)
                
            fps = FPS().start()
                
            while True:
                
                frame = camera.read()
                    
                if process_current_frame:
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                    # frame = np.dstack([frame, frame, frame])
                    frame_small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                    
                    # encoding = asyncio.create_task(detect_faces(frame_small))
                    self.face_locations, self.faces_names = self.detect_faces(frame_small)
                    

                self.process_current_frame = not self.process_current_frame

                for (top, right, bottom, left), name in zip(self.face_locations, self.faces_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                        
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left, bottom ), font, 1.0, (255, 255, 255), 1)
                    
                    
                    cv2.imshow('Video ', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                fps.update()
                
                # camera.release()
            camera.stop()
            cv2.destroyAllWindows()
        

if __name__ == "__main__":
    
    
    rtmp_url = "rtmp://192.168.0.25:1935/webcam/teste"

    ip = 'rtsp://192.168.0.35:5555/h264.sdp'
    print("iniciando a captura de imagens... ")
    # asyncio.run(gen_frames())
    fr = Faces_Recognition()
    fr.gen_frames()
    
   
    
