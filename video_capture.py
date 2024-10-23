import cv2 as cv
import face_recognition
import threading

rtmp_url = "rtmp://192.168.0.179:1935/small/teste"

camera = cv.VideoCapture(0)

all_faces_locations = []
all_faces_encoding = []

def encoding_frame(frame, face_locations):

    if face_locations:

        all_faces_encoding = face_recognition.face_encodings(frame, face_locations)
        print("aqui")

    return


while True:

    let,  frame = camera.read()

    frame_small =  cv.resize(frame, (0,0), fx=0.25, fy=0.25)
    
    all_faces_locations = face_recognition.face_locations(frame_small, number_of_times_to_upsample=1, model="hog")
    
    threading.Thread(target=encoding_frame, args=(frame_small, all_faces_locations)).start()
    
    for index, face in enumerate(all_faces_locations):
        top, right, bottom, left = face
        # threading.Thread(target=encoding_frame, args=(frame_small, all_faces_locations)).start()
        top *= 4
        right*= 4
        left*= 4
        bottom *= 4
        print(f'found  face {index+1} in location {top=}, {right=}, {bottom=} and {left=} ')
        cv.rectangle(frame,(left, top), (right, bottom), (0,0,255), 2) 
    cv.imshow("webcam Video ", frame)
    
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
           
camera.release()        
cv.destroyAllWindows()    