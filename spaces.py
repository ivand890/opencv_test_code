import cv2 as cv

# BGR --> Blue, Green, Red, OpenCV default color space
image_bgr = cv.imread("./Resources/Photos/park.jpg")
cv.imshow("BGR", image_bgr)

# Gray scale
image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", image_gray)

image_gray2bgr = cv.cvtColor(image_gray, cv.COLOR_GRAY2BGR)
image_gray2bgr2hsv = cv.cvtColor(image_gray2bgr, cv.COLOR_BGR2HSV)

cv.imshow("Gray2BGR2HSV", image_gray2bgr2hsv)


# HSV --> Hue, Saturation, Value(Brightness)
imahe_hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
cv.imshow("HSV", imahe_hsv)

# RGB --> Red, Green, Blue
image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
cv.imshow("RGB", image_rgb)

# Lab
image_lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2LAB)
cv.imshow("Lab", image_lab)

cv.waitKey(0)
cv.destroyAllWindows()
