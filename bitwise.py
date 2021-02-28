import cv2 as cv
import numpy as np

# blank image, rectangle and circle
blank = np.zeros((400, 400), dtype="uint8")
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)
cv.imshow("Rectangle", rectangle)
cv.imshow("Circle", circle)

# bitwise AND --> intersecions only
and_image = cv.bitwise_and(rectangle, circle)
cv.imshow("Bitwise AND", and_image)

# bitwise OR --> intersection and non-intersectios
or_image = cv.bitwise_or(rectangle, circle)
cv.imshow("Bitwise OR", or_image)

# bitwise XOR --> non-intersections only
xor_image = cv.bitwise_xor(rectangle, circle)
cv.imshow("Bitwise XOR", xor_image)

# bitwise NOT -->  binary invertion
not_image = cv.bitwise_not(rectangle)
cv.imshow("Bitwise NOT", not_image)

cv.waitKey(0)
cv.destroyAllWindows()
