import numpy as np
import cv2 as cv
import os

def makePredict(img, labels, recognition_model, haar_model, scale=1.1, minNeig=5):
    image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_rec = haar_model.detectMultiScale(image, scaleFactor=scale, minNeighbors=minNeig)
    for (x,y,w,h) in face_rec:
        roi = image[y:y+h, x:x+w]
        predic, acc = recognition_model.predict(roi)
        cv.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 2)
        cv.putText(img, f'{labels[predic]}', (x-5,y-5), cv.FONT_HERSHEY_SIMPLEX, .5, (0,225,0), 2)
        cv.putText(img, f'Acc: {round(acc, 2)}', (x+5,y+h-5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,225,0), 1)
    return img


#### test

haar_face = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face_recognicer = cv.face.LBPHFaceRecognizer_create()
face_recognicer.read('./LBPH_trained.yml')

names = os.listdir('./Resources/Faces/val')
names2 = os.listdir('./Resources/Faces/train')
print(names)
print(names2)
image = cv.imread('./Resources/Faces/val/Jerry Seinfield/3.jpg')
predicted = makePredict(image, names, face_recognicer, haar_face)
cv.imshow('Predicted', predicted)
cv.waitKey(0)
cv.destroyAllWindows()