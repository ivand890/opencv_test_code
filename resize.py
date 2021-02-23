import cv2 as cv
import numpy as np

def resizeFrame(frame, dim=None, scale=None, inter=cv.INTER_AREA):
    """Change scale of images, videos and live videos.

    Args:
        frame (numpy.ndarray): Image frame
        scale (float, optional): Output scale. Defaults to 0.75.

    Returns:
        numpy.ndarray: Scaled image frame
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
    return int(frame.shape[1]*(fraction)), int(frame.shape[0]*(fraction))

