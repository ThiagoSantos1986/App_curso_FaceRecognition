from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import face_recognition
import cv2

def filterFrame(frame):
	frame = imutils.resize(frame, width=450)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = np.dstack([frame, frame, frame])
	return frame


rtmp_url = "rtmp://192.168.0.25:1935/small/teste"

print("[INFO] starting video file thread...")
fvs = FileVideoStream(path=rtmp_url, transform=filterFrame).start()
time.sleep(1.0)

# start the FPS timer
fps = FPS().start()

# loop over frames from the video file stream
while fvs.running():
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale (while still retaining 3
	# channels)
    frame = fvs.read()
    
    frame_small =  cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    
    
    all_faces_locations = face_recognition.face_locations(frame_small, number_of_times_to_upsample=2, model="hog")
    
    for index, face in enumerate(all_faces_locations):
        
        top, right, bottom, left = face
        
        top *= 4
        right *= 4
        left *= 4
        bottom *= 4

        print(f'found  face {index+1} in location {top=}, {right=}, {bottom=} and {left=} ')

        

        cv2.rectangle(frame,(left, top), (right, bottom), (0,200,0), 2) 
	# show the frame and update the FPS counter
    cv2.imshow("Frame", frame)

    cv2.waitKey(1)
    if fvs.Q.qsize() < 2:  # If we are low on frames, give time to produce
        time.sleep(0.001)  # Ensures producer runs now, so 2 is sufficient
    fps.update()
  
    
    
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()