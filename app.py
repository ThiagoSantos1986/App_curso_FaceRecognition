import cv2 as cv
import face_recognition


img_read = cv.imread('two_people.jpg')



all_face_location = face_recognition.face_locations(img_read, model='hog')
print(f'number the face: {len(all_face_location)}')


for index, face in enumerate(all_face_location):
    top, right, bottom, left = face
    print(f'found  face {index+1} in location {top=}, {right=}, {bottom=} and {left=} ')

    img_face = img_read[top:bottom, left:right]

    cv.imshow(f'Face {index+1}', img_face)
cv.waitKey(0)
cv.destroyAllWindows()




