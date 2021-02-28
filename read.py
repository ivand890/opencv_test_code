import numpy as np
import cv2 as cv
from resize import resizeFrame


# cap = cv.VideoCapture('./Resources/Photos/cat_large.jpg')
# ret, frame = cap.read()
# frame = rescaleFrame(frame, scale=0.5)
# print(type(cap))
# cv.imshow('frame', frame)


cap = cv.VideoCapture(0)
# changeRes(cap, 100, 100)
while True:
    ret, frame = cap.read()
    frame = resizeFrame(frame, scale=0.5)
    cv.imshow("frame", frame)

    if cv.waitKey(1) == ord("q"):
        break

# cv.waitKey(0)
cap.release()
cv.destroyAllWindows()
