import os
import cv2 as cv

# haar cascade clasifier
haar_face = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_rect = haar_face.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in face_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("frame", frame)

    if cv.waitKey(1) == ord("q"):
        break

# cv.waitKey(0)
cap.release()
cv.destroyAllWindows()
