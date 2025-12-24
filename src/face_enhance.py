import cv2
import numpy as np
from gfpgan import GFPGANer


class FaceEnhancer:
    def __init__(self):
        self.gfpgan = GFPGANer(
            model_path="models/gfpgan/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device="cpu"
        )

    def enhance_face(self, image, bbox):
        """
        STEP 9.1 + STEP 9.4
        - Natural face blending
        - Full safety guards
        - Identity preservation
        """

        # ---------- GUARD 1 ----------
        if bbox is None:
            return image

        x, y, w, h = bbox
        img_h, img_w, _ = image.shape

        # ---------- GUARD 2: SAFE CLIPPING ----------
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            return image

        # ---------- FACE CROP ----------
        face_crop = image[y:y + h, x:x + w].copy()

        # ---------- GFPGAN INFERENCE ----------
        try:
            _, _, restored_faces = self.gfpgan.enhance(
                face_crop,
                has_aligned=False,
                only_center_face=True,
                paste_back=False
            )
        except Exception as e:
            print("[GFPGAN ERROR]:", e)
            return image

        # ---------- GUARD 3 ----------
        if restored_faces is None or len(restored_faces) == 0:
            return image

        restored_face = restored_faces[0]

        # ---------- SIZE CONSISTENCY ----------
        restored_face = cv2.resize(restored_face, (w, h))

        # ---------- STEP 9.1: NATURAL BLENDING ----------
        # Create soft alpha mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (w // 2, h // 2),
            (w // 2 - 5, h // 2 - 5),
            0, 0, 360,
            255,
            -1
        )

        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask = mask.astype(float) / 255.0
        mask = cv2.merge([mask, mask, mask])

        # Blend restored face with original
        blended_face = (
            restored_face.astype(float) * mask +
            face_crop.astype(float) * (1 - mask)
        ).astype(np.uint8)

        # ---------- STEP 9.4: FINAL SAFE PASTE ----------
        image[y:y + h, x:x + w] = blended_face

        return image
