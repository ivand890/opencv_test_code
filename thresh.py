import cv2 as cv

# Original image
image = cv.imread("./Resources/Photos/cats.jpg")
cv.imshow("Original", image)

# Gray Scale
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray_image)

# simple threshold
threshold, thresh_image = cv.threshold(gray_image, 150, 255, cv.THRESH_BINARY)
cv.imshow("Simple Thresh", thresh_image)

# simple threshold inverted
threshold, thresh_inv_image = cv.threshold(gray_image, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow("Simple Inv Thresh", thresh_inv_image)

# Adaptative threshold
adapt_mean_thresh = cv.adaptiveThreshold(
    gray_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 1
)
cv.imshow("Adaptative Threshold Mean", adapt_mean_thresh)

adapt_gauss_thresh = cv.adaptiveThreshold(
    gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 1
)
cv.imshow("Adaptative Threshold Gauss", adapt_gauss_thresh)


cv.waitKey(0)
cv.destroyAllWindows()
