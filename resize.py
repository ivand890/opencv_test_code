import cv2 as cv
import numpy as np


def resizeFrame(frame, dim=None, scale=None, inter=cv.INTER_AREA):
    """Resize images or frames

    Args:
        frame (ndarray): image to resize`
        dim (tuple, optional): Destination dimensions. Defaults to None.
        scale (float, optional): Destination scale. Defaults to None.
        inter (cv.INTER, optional): Interpolation function. Defaults to cv.INTER_AREA.

    Returns:
        ndarray: Resised frame
    """
    if dim and not scale:
        return cv.resize(frame, dim, interpolation=inter)
    elif dim and scale:
        width = int(dim[1] * scale)
        height = int(dim[0] * scale)
        dimension = (width, height)
        return cv.resize(frame, dimension, interpolation=inter)
    elif scale:
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimension = (width, height)
        return cv.resize(frame, dimension, interpolation=inter)
    else:
        return frame


def fractionFrames(frame, fraction):
    """Calculate the fraction dimension of a frame

    Args:
        frame (numpy.ndarray): Image frame
        fraction (float): fraction to compute

    Returns:
        (width, heigth): fraction for dimension
    """
    return int(frame.shape[1] * (fraction)), int(frame.shape[0] * (fraction))

