import cv2 as cv
import numpy as np

# Original
image = cv.imread("./Resources/Photos/cats.jpg")
cv.imshow("Original", image)

# blank
blank_img = np.zeros(image.shape, dtype="uint8")
blank_img2 = np.zeros(image.shape, dtype="uint8")

# Gray
gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray_img)

# blur
blur_img = cv.GaussianBlur(gray_img, (5, 5), cv.BORDER_DEFAULT)
cv.imshow("Blur", blur_img)

# Canny
canny_img = cv.Canny(blur_img, 125, 175)
cv.imshow("Canny", canny_img)

# threshold
ret, thresh_img = cv.threshold(blur_img, 125, 255, cv.THRESH_BINARY)
cv.imshow("Thresh", thresh_img)

# contour
contours_canny, hierarchies_canny = cv.findContours(
    canny_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
)
contours_thresh, hierarchies_thresh = cv.findContours(
    thresh_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
)
print(f"{len(contours_canny)} --> canny\n{len(contours_thresh)} --> thresh")

cv.drawContours(blank_img, contours_canny, -1, (0, 0, 255), 1)
cv.drawContours(blank_img2, contours_thresh, -1, (0, 0, 255), 1)

cv.imshow("Contours_canny", blank_img)
cv.imshow("Contours_thresh", blank_img2)

cv.waitKey(0)
cv.destroyAllWindows()
