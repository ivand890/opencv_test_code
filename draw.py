import cv2 as cv
import  numpy as np
from resize import fractionFrames

# base image ***B G R***
img = np.zeros((500, 500, 3), dtype=np.uint8)
#img[:] = 255, 0, 0
cv.imshow('base', img)

# draw line
img = cv.line(img, (0, 0), (img.shape[1], img.shape[0]), (255, 0, 0), 1)
cv.imshow('line', img)

# draw rectangle
# img = cv.rectangle(img, (img.shape[1]//3, img.shape[0]//3), (int(img.shape[1]*(2/3)), int(img.shape[0]*(2/3))), (0, 255, 0), 3)
img = cv.rectangle(img, fractionFrames(img, 1/3), fractionFrames(img, 2/3), (0, 255, 0), 3)
cv.imshow('rect', img)

#draw circle
img = cv.circle(img, fractionFrames(img, 1/2), 25, (0, 0, 255), -1)
cv.imshow('circle', img)

#draw text

img = cv.putText(img, 'Hello OpenCV!!', (5, fractionFrames(img, 9/10)[0]), cv.FONT_HERSHEY_SIMPLEX, 2, (122, 122, 122), 1)
cv.imshow('txt', img)

cv.waitKey(0)
cv.destroyAllWindows()