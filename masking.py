import cv2 as cv
import numpy as np
import resize as rze

# Original image
image = cv.imread("./Resources/Photos/cats 2.jpg")
cv.imshow("Original", image)

# blank image, base mask
blank = np.zeros(image.shape[:2], dtype="uint8")
rectangle = cv.rectangle(
    blank.copy(),
    (rze.fractionFrames(blank, 1 / 3)),
    (rze.fractionFrames(blank, 2 / 3)),
    255,
    -1,
)
circle = cv.circle(
    blank.copy(),
    (rze.fractionFrames(blank, 1 / 2)),
    rze.fractionFrames(blank, 1 / 2)[0] - rze.fractionFrames(blank, 1 / 3)[0],
    255,
    -1,
)
mask = cv.bitwise_or(rectangle, circle)
cv.imshow("Rectangle", rectangle)
cv.imshow("Circle", circle)
cv.imshow("Mask", mask)

# Masking
masked_image = cv.bitwise_and(image, image, mask=mask)
cv.imshow("Masked", masked_image)

cv.waitKey(0)
cv.destroyAllWindows()
