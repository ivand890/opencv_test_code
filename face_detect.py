import os
import cv2 as cv

# Original image
image = cv.imread("./Resources/Photos/group 2.jpg")
cv.imshow("Original", image)

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray_image)

# haar cascade clasifier
haar_face = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face_rect = haar_face.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in face_rect:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imshow("Faces", image)


cv.waitKey(0)
cv.destroyAllWindows()
