import os
import cv2 as cv
import numpy as np


def create_train_data(path, detec_model, scale=1.1, minNeig=5):
    peoples = os.listdir(path)
    features, labels = [], []
    for person in peoples:
        person_data = os.listdir(os.path.join(path, person))
        for face in person_data:
            image = cv.cvtColor(
                cv.imread(os.path.join(path, person, face)), cv.COLOR_BGR2GRAY
            )
            face_detec = detec_model.detectMultiScale(
                image, scaleFactor=scale, minNeighbors=minNeig
            )
            for (x, y, w, h) in face_detec:
                roi_image = image[y : y + h, x : x + w]
                features.append(roi_image)
                labels.append(peoples.index(person))
    return np.array(features, dtype="object"), np.array(labels)


TRAIN_DIR = "./Resources/Faces/train"
haar_face = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

f, l = create_train_data(TRAIN_DIR, haar_face)
print(f"features: {len(f)}, labels: {len(l)}")

face_recognicer = cv.face.LBPHFaceRecognizer_create()
face_recognicer.train(f, l)
face_recognicer.save("LBPH_trained.yml")

