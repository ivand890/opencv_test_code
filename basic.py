import cv2 as cv

# reading image
image = cv.imread('./Resources/Photos/park.jpg')
cv.imshow('original', image)

# convert to gray scale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# bluring image
blur = cv.GaussianBlur(image, (7, 7), cv.BORDER_DEFAULT)
cv.imshow('GBlur', blur)

#edge detection
edge = cv.Canny(blur, 120, 180)
cv.imshow('Edges', edge)

#dilating de image
dilated = cv.dilate(edge, (7, 7 ), iterations=3)
cv.imshow('Dilated', dilated)

# eroding
eroded = cv.erode(dilated, (7, 7), iterations=3)
cv.imshow('Eroded', eroded)

#resize
resized = cv.resize(image, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow('Resized', resized)

#cropping
cropped = image[200:300, 200:300]
cv.imshow('Crop', cropped)

cv.waitKey(0)
cv.destroyAllWindows()