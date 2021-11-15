#!/usr/bin/env python3
from cv2 import cv2
#from playsound import playsound


cascPath = "/home/tage/Documents/Projects/Automatic-Doorbell/local/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

vid = cv2.VideoCapture(-1)

while(True):

    ret ,frame = vid.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30,30)) 

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    #playsound('/Documents/Projects/Automatic-Doorbell/local/lyd.mp3')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
