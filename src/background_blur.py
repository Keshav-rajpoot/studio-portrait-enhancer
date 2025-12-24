import cv2
import mediapipe as mp
import numpy as np


class BackgroundBlur:
    def __init__(self, blur_strength=35):
        self.blur_strength = blur_strength
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

    def apply(self, image):
        """
        Apply background blur while keeping person sharp
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(rgb)

        if results.segmentation_mask is None:
            return image

        mask = results.segmentation_mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        condition = mask > 0.5

        blurred_bg = cv2.GaussianBlur(
            image,
            (self.blur_strength, self.blur_strength),
            0
        )

        output = np.where(condition[..., None], image, blurred_bg)
        return output
