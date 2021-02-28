import cv2 as cv
import numpy as np

# Original image
image = cv.imread("./Resources/Photos/park.jpg")
cv.imshow("Original", image)

# split color channels
b, g, r = cv.split(image)
cv.imshow("Blue", b)
cv.imshow("Green", g)
cv.imshow("Red", r)

# merge color channes
image_bgr = cv.merge([b, g, r])
cv.imshow("BGR", image_bgr)

# color channesl whit color
blank = np.zeros(image.shape[:2], dtype="uint8")
image_blue = cv.merge([b, blank, blank])
image_green = cv.merge([blank, g, blank])
image_red = cv.merge([blank, blank, r])
cv.imshow("Img_Blue", image_blue)
cv.imshow("Img_Green", image_green)
cv.imshow("Img_Red", image_red)

cv.waitKey(0)
cv.destroyAllWindows()
