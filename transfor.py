import cv2 as cv
import numpy as np

# reading image
image = cv.imread("./Resources/Photos/park.jpg")
cv.imshow("original", image)

# translate


def translate(img, x, y):
    """Translate imgage in x, y directions,
        x  --> Right
        -x --> Left
        y  --> Down
        -y --> Up

    Args:
        img (ndarray): Image to translate
        x (int8): offset in x direction
        y (int8): offset in Y direction

    Returns:
        ndarray: Translated image
    """
    transtMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = tuple([img.shape[1], img.shape[0]])
    return cv.warpAffine(img, transtMat, dimensions)


trans_img = translate(image, 0, -100)
cv.imshow("Translate", trans_img)

# rotations


def rotate(img, angle, rotPoint=None):
    """Rotate image in desired angle and ritation point.
        angle --> counterclockwise rotations
        -ange --> clockwise rotations

    Args:
        img (ndarray): Image to rotate
        angle (float): Rotation angle in degees
        rotPoint (tuple, optional): Rotation point. Defaults to image's center.

    Returns:
        ndarray: Rotated image
    """
    height, width = img.shape[:2]
    if rotPoint == None:
        rotPoint = tuple([width // 2, height // 2])
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    return cv.warpAffine(img, rotMat, (width, height))


rot_img = rotate(image, -90)
cv.imshow("Rotate", rot_img)

# Resize
resize_img = cv.resize(image, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow("Resize", resize_img)

# Fliping
flip_img = cv.flip(image, -1)  # 0 --> vertical; 1 --> horizontal; -1 --> both of them
cv.imshow("Flip", flip_img)

# Cropping
crop_img = image[100:200, 200:500]
cv.imshow("Cropped", crop_img)

cv.waitKey(0)
cv.destroyAllWindows()
