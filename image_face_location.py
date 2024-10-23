import face_recognition as fr
import cv2


#recurepando foto para read
foto_original = cv2.imread('images/testing/trump-modi-unknown.jpg')

trump_load = fr.load_image_file('images/samples/trump.jpg')
trump_enconding = fr.face_encodings(trump_load)[0]


modi_load = fr.load_image_file('images/samples/modi.jpg')
modi_enconding = fr.face_encodings(modi_load)[0]

knows_names = ['trump', 'modi']
knows_faces_encoding = [trump_enconding, modi_enconding]

#carregando  a foto para leitura 
image_to_recognize = fr.load_image_file('images/testing/trump-modi-unknown.jpg')

#localizando as posicoes dos rostos na foto
all_faces_location = fr.face_locations(image_to_recognize,number_of_times_to_upsample=1, model="hog")
all_faces_encondings = fr.face_encodings(image_to_recognize, all_faces_location)

# print(f'total de rostos : {len(all_faces_location)}')




for current_face_location, current_face_encoding in zip(all_faces_location, all_faces_encondings):
   
    top_pos, right_pos, bototm_pos, left_pos = current_face_location
    
    all_matches = fr.compare_faces(knows_faces_encoding, current_face_encoding)

    name_of_person = 'Unknown face'

    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = knows_names[first_match_index]

    cv2.rectangle(foto_original, (left_pos, top_pos), (right_pos, bototm_pos), (255,0,0 ), 1)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(foto_original, name_of_person, (left_pos, bototm_pos), font, 0.5, (255,0,0),1)

cv2.imshow("faces reconhecidas", foto_original)

if cv2.waitKey(10000) & 0xFF == ord('q'):
            
    cv2.destroyAllWindows()
    