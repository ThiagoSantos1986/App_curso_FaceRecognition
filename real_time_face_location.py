import face_recognition as fr
import logging
from logging import INFO, basicConfig, WARNING, DEBUG
from logging import critical,info, warning
import cv2
import numpy as np
import threading
import dlib



basicConfig(level=DEBUG,
                        filename='TESTANDO.log',
                        filemode='w',
                        format='%(levelname)s: %(name)s: %(asctime)s: %(message)s'
            )

logger = logging.getLogger('simple_example')


camera = cv2.VideoCapture(0)


trump_load = fr.load_image_file('images/samples/trump.jpg')
trump_enconding = fr.face_encodings(trump_load)[0]


modi_load = fr.load_image_file('images/samples/modi.jpg')
modi_enconding = fr.face_encodings(modi_load)[0]

knows_names = ['trump', 'modi']
knows_faces_encoding = [trump_enconding, modi_enconding]

all_faces_locations = []
all_faces_encoding = []
all_faces_names = []
process_this_frame = True
while True:

    let,  frame = camera.read()

    if process_this_frame:

        current_frame_small =  cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        
        # rgb_small_frame = current_frame_small[:, :, ::-1]
        
        all_faces_locations = fr.face_locations(current_frame_small, number_of_times_to_upsample=1, model="hog")

        all_faces_encoding = fr.face_encodings(current_frame_small, all_faces_locations)


        all_faces_names = []

        for encoding in all_faces_encoding:

            all_matches = fr.compare_faces(knows_faces_encoding, encoding)

            name_of_person = 'Unknown face'

            face_distances = fr.face_distance(knows_faces_encoding, encoding)
            # print("The matching percentage is {} against the sample {}".format(round(((1-float(face_distance))*100),2),known_face_names[i]))
            best_match_index = np.argmin(face_distances)
            if all_matches[best_match_index]:
                
                name_of_person = knows_names[best_match_index]
                
                logger.info(f'Rosto Localizado--> {name_of_person.upper()}')

            else:
                warning('Desconhecido no local')
            
            all_faces_names.append(name_of_person)
    
    process_this_frame = not process_this_frame

    for (top_pos, right_pos, bottom_pos, left_pos), name in zip(all_faces_locations, all_faces_names):
    
        # top_pos, right_pos, bottom_pos, left_pos = current_face_location

        top_pos *= 4
        right_pos *= 4
        left_pos *= 4
        bottom_pos *= 4
        
        cv2.rectangle(frame, (left_pos, top_pos), (right_pos, bottom_pos), (255,0,0 ), 1)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left_pos, bottom_pos), font, 0.5, (255,0,0),1)

    cv2.imshow("faces reconhecidas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()




