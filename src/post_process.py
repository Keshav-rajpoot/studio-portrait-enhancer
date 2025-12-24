import cv2
import numpy as np


def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def enhance_sharpness(image):
    gaussian = cv2.GaussianBlur(image, (0, 0), 1.0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)


def post_process(image):
    image = enhance_contrast(image)
    image = enhance_sharpness(image)
    image = color_balance(image)
    return image


def color_balance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    a = cv2.subtract(a, 2)
    b = cv2.subtract(b, 2)

    balanced = cv2.merge((l, a, b))
    return cv2.cvtColor(balanced, cv2.COLOR_LAB2BGR)

def studio_polish(image):
    """
    Final aesthetic polish for studio look
    """
    # ---- Soft contrast curve ----
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ---- Mild skin-safe smoothing ----
    smooth = cv2.bilateralFilter(image, d=7, sigmaColor=30, sigmaSpace=30)

    # ---- Blend for natural look ----
    final = cv2.addWeighted(image, 0.85, smooth, 0.15, 0)

    return final
