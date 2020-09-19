import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('imagesBasic/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('imagesBasic/Bill Gates.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

result = face_recognition.compare_faces([encodeElon], encodeTest)
faceDist = face_recognition.face_distance([encodeElon], encodeTest)
cv2.putText(imgTest, f'{result}{round(faceDist[0],2)}', (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 240, 240), 1)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
