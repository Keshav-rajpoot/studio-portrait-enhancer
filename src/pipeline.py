import cv2
import time

from src.face_detect import detect_face
from src.face_enhance import FaceEnhancer
from src.background_blur import BackgroundBlur
from src.post_process import post_process


class StudioPortraitPipeline:
    def __init__(self):
        self.face_enhancer = FaceEnhancer()
        self.bg_blur = BackgroundBlur()

    def process(self, image):
        """
        Full studio portrait enhancement pipeline
        """
        start_time = time.time()

        # 1. Face Detection
        bbox = detect_face(image)

        # 2. Face Enhancement (GFPGAN)
        if bbox is not None:
            image = self.face_enhancer.enhance_face(image, bbox)

        # 3. Background Blur (Portrait Bokeh)
        image = self.bg_blur.apply(image)

        # 4. Final Post Processing
        image = post_process(image)

        end_time = time.time()
        print(f"Inference Time: {end_time - start_time:.2f}s")

        return image
