
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

def face_distances_percent(face_distances, face_match_media=0.6):
    range = (1.0 - face_match_media) 
    linear_value = (1.0 - face_distances) / (range * 2.0)
    
    if face_distances > face_match_media:
        return str(round(linear_value * 100, 2) + '%')
    else:
        value = (linear_value + ((1.0 -linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

def filterFrame(frame):
    
    frame = imutils.resize(frame, width=500)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.flip(frame, 1) 
    frame = np.dstack([frame, frame, frame])
    
    return frame


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
    
    # def encoding_faces(self):
        
    #     # face_data_base = Encoding.query.order_by(Encoding.encoding.desc()).all()
    
    #     for face in face_data_base:
            
    #         query_pessoa = Pessoa.query.filter_by(id_pessoa=face.id_pessoa).first() #pega a pessoa
    #         self.known_face_names.append(query_pessoa.nome)
                
    #         record = face.encoding
    #         unpack_b5 = np.frombuffer(record, dtype=np.float64)
    #         self.known_face_encodings.append(unpack_b5)
        
    #     print(self.known_face_names)           
        
      
        
    def run_recogntion(self):
        
        # camera = cv2.VideoCapture(self.src)
        # camera = WebcamVideoStream().start()
        camera = VideoStream().start()
        
        time.sleep(1.0)
    
        fps = FPS().start()
        
        # if not video_capture.isOpened():
        #     sys.exit('Video source not found')
        
        
        while True:
            frame = camera.read()
            
            if self.process_current_frame:
               
                
                # frame = np.dstack([frame, frame, frame])
                # frame = filterFrame(frame)
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = np.dstack([frame, frame, frame])
                frame_small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                
                
                self.face_location = face_recognition.face_locations(frame_small, number_of_times_to_upsample=1)           
                self.face_encoding = face_recognition.face_encodings(frame_small, self.face_location)
                
                self.face_names = []
                
                for face_encoding in self.face_encoding:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = "Unknown"
                    
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        # confidence = face_distances_percent(face_distances[best_match_index])
                        
                    self.face_names.append(f'{name}')
                      
            self.process_current_frame = not self.process_current_frame   
        
            for (top, right, bottom, left), name in zip(self.face_location, self.face_names):     
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
            
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 1)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            
            _ , jpeg = cv2.imencode('.jpg', frame)
        
            frame  = jpeg.tobytes()
            
            yield (b'--frame\r\n' b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            fps.update()
        #     cv2.imshow('Video ', frame)
        
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
            # fps.update()
        # camera.stop()
        # video_capture.release()
        # cv2.destroyAllWindows()


    def detect_faces(self, frame):
        
        # frame = imutils.resize(frame, width=500)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.dstack([frame, frame, frame])

        # frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # frame = cv2.flip(frame, 1) 
        
        frame_small = cv2.resize(frame, (0,0), fx=self.frame_resizing, fy=self.frame_resizing)
                
        # frame_small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
               
                
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
                warning("pessoa desconhecida no local..")
                    
            self.faces_names.append(name)        
                
                    

        # self.face_location = np.array(self.face_location)
        # self.face_location = self.face_location / self.frame_resizing
        return self.face_location , self.faces_names
        
    
    def gen_webcam(self, ip=0):
        logger = logging.getLogger('Mike faces')
        
        process_current_frame = True
        
        # camera = WebcamVideoStream().start()
        camera = VideoStream(src=ip).start()
        
        time.sleep(1.0)
        
        fps = FPS().start()
    
    
        
        while True:
        
            frame = camera.read()
            
            if process_current_frame:
            
                # frame = filterFrame(frame)
                 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # frame = cv2.flip(frame, 1) 
                frame = np.dstack([frame, frame, frame])
        
                
                frame_small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                
                self.face_locations = face_recognition.face_locations(frame_small, number_of_times_to_upsample=1, model='hog')

                self.face_encodings = face_recognition.face_encodings(frame_small, self.face_locations)
                
                self.faces_names = []
                for encoding in self.face_encodings:
                    
                    matches = face_recognition.compare_faces(self.known_face_encodings, encoding, tolerance=0.6)
                        
                    name = "Unknown"
                    # print(name)
                    if True in matches:
                        first_match_index = matches.index(True)              
                        name = self.known_face_names[first_match_index]  
                        logger.info(f'Rosto Localizado {name}')
                        
                    else:
                        logger.warning("pessoa desconhecida no local..")
                    
                    self.faces_names.append(name)

            self.process_current_frame = not self.process_current_frame  
            
            for (top, right, bottom, left), name in zip(self.face_locations, self.faces_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left, bottom ), font, 1.0, (255, 255, 255), 1)
            
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break
            cv2.imshow('Video ', frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.update()
        
        # camera.release()
        camera.stop()
        cv2.destroyAllWindows()
            # _ , jpeg = cv2.imencode('.jpg', frame)
            # frame  = jpeg.tobytes()
            
            # yield (b'--frame\r\n' b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            # fps.update()
            
    def gen_frames(self, ip=0):
        
        process_current_frame = True
            
        # camera = WebcamVideoStream().start()
        camera = VideoStream(src=ip).start()
            
        time.sleep(1.0)
            
        fps = FPS().start()
            
        while True:
            
            frame = camera.read()
                
            if process_current_frame:
                self.face_locations, self.faces_names = self.detect_faces(frame)

            self.process_current_frame = not self.process_current_frame  
                
            for (top, right, bottom, left), name in zip(self.face_locations, self.faces_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                    
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left, bottom ), font, 1.0, (255, 255, 255), 1)
                
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     break
                cv2.imshow('Video ', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fps.update()
            
            # camera.release()
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
    fr = Face_Recognition()
    fr.gen_frames()
    # fr.gen_webcam(ip)
    
