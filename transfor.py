import cv2 as cv

# reading image
image = cv.imread('./Resources/Photos/park.jpg')
cv.imshow('original', image)