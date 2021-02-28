import cv2 as cv

# Original image

image = cv.imread("./Resources/Photos/cats.jpg")
cv.imshow("Original", image)

# Average Blur
avgblur_image = cv.blur(image, (3, 3))
cv.imshow("AVG Blur", avgblur_image)

# Gaussian Blur
gaublur_image = cv.GaussianBlur(image, (3, 3), 0)
cv.imshow("Gaussian Blur", gaublur_image)

# Median Blur
medianblur_image = cv.medianBlur(image, 3)
cv.imshow("Median Blur", medianblur_image)

# Bilateral Blur
bilateralblur_image = cv.bilateralFilter(image, 10, 35, 25)
cv.imshow("Bilateral Blur", bilateralblur_image)

cv.waitKey(0)
cv.destroyAllWindows()
