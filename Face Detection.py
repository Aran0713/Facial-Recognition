from tkinter import Image
import cv2
import numpy as np
import face_recognition
import os

path = '../Face Images'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)

# Getting images
for cl in myList:
    currentImg = cv2.imread(f'{path}/{cl}')
    images.append(currentImg)
    classNames. append(os.path.splitext(cl)[0])
#print(classNames)

def findEncodings(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return encodedList

encodedList = findEncodings(images)
print("Encoding complete")


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgFrame = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgFrame =  cv2.cvtColor(imgFrame, cv2.COLOR_BGR2RGB)
    
    faceLocationsFrame = face_recognition.face_locations(imgFrame)
    faceEncodedFrame = face_recognition.face_encodings(imgFrame, faceLocationsFrame)
    
    for encodedFace, faceLocation in zip(faceEncodedFrame, faceLocationsFrame):
        matches = face_recognition.compare_faces(encodedList, encodedFace)
        ratings = face_recognition.face_distance(encodedList, encodedFace)
       
        matchIndex = np.argmin(ratings)
        print(ratings[matchIndex])
        
        if ratings[matchIndex]:
            y1,x2,y2,x1 = faceLocation
            y1,x2,y2,x1 = y1*4 -30, x2*4 +10, y2*4 +30, x1*4 -10
            
            cv2.rectangle(img, (x1,y1), (x2,y2), (255, 127, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2,y2), (255, 127, 0), cv2.FILLED)
            if ratings[matchIndex] < 0.6:
                name = classNames[matchIndex].upper()
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            else:
                cv2.putText(img, "Unknown", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == 13:
        break
    
cv2.destroyAllWindows()
        






