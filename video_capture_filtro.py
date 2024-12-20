import cv2 as cv
import face_recognition
import imutils

rtmp_url = "rtmp://192.168.0.179:1935/small/teste"

camera = cv.VideoCapture(0)

all_faces_locations = []


while True:

    let,  frame = camera.read()

    frame_small =  cv.resize(frame, (0,0), fx=0.25, fy=0.25)
    all_faces_locations = face_recognition.face_locations(frame_small, number_of_times_to_upsample=2, model="hog")
    
    for index, face in enumerate(all_faces_locations):
        
        top, right, bottom, left = face
        
        top *= 4
        right *= 4
        left *= 4
        bottom *= 4

        print(f'found  face {index+1} in location {top=}, {right=}, {bottom=} and {left=} ')

        # pegando  o frame do video e recuperando a posicao do rosto em tempo real
        currente_face =  frame[top:bottom, left:right]
        #pegamos a posicao do rosto e fazemos o efeito na imagem
        currente_face = cv.GaussianBlur(currente_face, (99,99), 30)
        # substiruindo a face atual pela imagem com efeito
        frame[top:bottom, left:right] = currente_face

        cv.rectangle(frame,(left, top), (right, bottom), (0,200,0), 2) 
    cv.imshow("webcam Video ", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
           
camera.release()        
cv.destroyAllWindows()    