import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection


def detect_face(image):
    h, w, _ = image.shape

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.6
    ) as face_detection:

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if not results.detections:
            return None

        bbox = results.detections[0].location_data.relative_bounding_box

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        return x, y, bw, bh
