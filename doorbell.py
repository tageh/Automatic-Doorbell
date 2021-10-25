#!/usr/bin/env python3
import sys
from playsound import playsound
from cv2 import cv2

#cascPath = sys.argv[1]
cascPath = "/usr/local/lib/python3.9/dist-packages/cv2/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

vid = cv2.VideoCapture(-1)

while(True):

    ret ,frame = vid.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30,30)) 

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        playsound("Doorbell-SoundBible.com-516741062.mp3")

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
