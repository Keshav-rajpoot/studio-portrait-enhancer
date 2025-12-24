import cv2
import numpy as np


def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def remove_motion_blur(image):
    """
    Lightweight deblurring (safe for portraits)
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    deblurred = cv2.filter2D(image, -1, kernel)
    return deblurred
