import cv2
import numpy as np
import face_recognition
import os

path = 'imagesAttendence'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings (images):
     encodingList = []
     for img in images:
         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
         encode = face_recognition.face_encodings(img)[0]
         encodingList.append(encode)
     return encodingList


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
    print("Could not open video device")

while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0, 25, 0, 25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgSmall)
    encodesCurrentFrame = face_recognition.face_encodings(imgSmall, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDist)
        matchesIndex = np.argmin(faceDist)

        if matches[matchesIndex]:
            name = classNames[matchesIndex].upper()
            print(name)
