import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np


def eval_model(path, recog_model, detec_model, scale=1.1, minNeig=5):
    peoples = os.listdir(path)
    predic_label, real_label, confidence = [], [], []
    for person in peoples:
        person_data = os.listdir(os.path.join(path, person))
        for face in person_data:
            image = cv.imread(os.path.join(path, person, face))
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            face_dect = detec_model.detectMultiScale(image_gray, scaleFactor=scale, minNeighbors=minNeig)
            for (x,y,w,h) in face_dect:
                roi_image = image_gray[y:y+h, x:x+w]
                p_label, conf = recog_model.predict(roi_image)
                predic_label.append(p_label)
                real_label.append(peoples.index(person))
                confidence.append(conf)
    return np.array(predic_label), np.array(real_label), np.array(confidence)

DIR_EVAL = './Resources/Faces/val'
haar_face = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_recognicer = cv.face.LBPHFaceRecognizer_create()
face_recognicer.read('./LBPH_trained.yml')

p, r, c = eval_model(DIR_EVAL, face_recognicer, haar_face)

plt.figure()
plt.plot(p, 'or')
plt.plot(r, 'xb')
plt.show()
plt.figure()
plt.plot(c)
plt.show()

abs_error = (p == r).astype(int)
accuracy = np.sum(abs_error)/np.size(abs_error)
print(f'accuracy: {accuracy}')