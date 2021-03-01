import cv2 as cv
import numpy as np

# Original image
image = cv.imread("./Resources/Photos/park.jpg")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Original", image)
cv.imshow("Gray", gray_image)

# Laplacian gradient

laplacian = cv.Laplacian(gray_image, cv.CV_64F)
laplacian_image = np.uint8(np.absolute(laplacian))
cv.imshow("Laplacian", laplacian_image)

# Sobel
sobelx_image = cv.Sobel(gray_image, cv.CV_64F, 1, 0)
sobely_image = cv.Sobel(gray_image, cv.CV_64F, 0, 1)
sobelxxyy_image = cv.Sobel(gray_image, cv.CV_64F, 1, 1)
sobelxy_image = cv.bitwise_or(sobelx_image, sobely_image)
cv.imshow("Sobel X", sobelx_image)
cv.imshow("Sobel Y", sobely_image)
cv.imshow("Sobel XY", sobelxy_image)
cv.imshow("Sobel XXYY", sobelxxyy_image)

# Canny
canny_image = cv.Canny(gray_image, 160, 175)
cv.imshow("Canny", canny_image)


cv.waitKey(0)
cv.destroyAllWindows()

