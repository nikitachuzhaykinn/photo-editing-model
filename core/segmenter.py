"""Сегментация: лицо, кожа, фон (MediaPipe Tasks API)."""
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
FACE_MODEL = os.path.join(MODELS_DIR, "face_landmarker.task")
SELFIE_MODEL = os.path.join(MODELS_DIR, "selfie_segmenter.tflite")

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


class FaceSegmenter:
    def __init__(self):
        face_opts = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)

        seg_opts = vision.ImageSegmenterOptions(
            base_options=python.BaseOptions(model_asset_path=SELFIE_MODEL),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
        )
        self.selfie = vision.ImageSegmenter.create_from_options(seg_opts)

    def _face_oval_mask(self, image):
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        pts = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in FACE_OVAL],
            dtype=np.int32,
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    @staticmethod
    def _skin_mask(image, face_mask):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 128, 70], dtype=np.uint8)
        upper = np.array([255, 180, 135], dtype=np.uint8)
        skin = cv2.inRange(ycrcb, lower, upper)
        skin = cv2.bitwise_and(skin, face_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=2)
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.GaussianBlur(skin, (11, 11), 0)

    def _background_mask(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.selfie.segment(mp_image)
        person = (result.confidence_masks[0].numpy_view() > 0.5).astype(np.uint8) * 255
        return cv2.bitwise_not(person)

    def build_masks(self, image):
        h, w = image.shape[:2]
        face = self._face_oval_mask(image)

        if face is None:
            empty = np.zeros((h, w), dtype=np.uint8)
            return {"face": empty, "skin": empty, "background": empty, "found": False}

        skin = self._skin_mask(image, face)
        background = self._background_mask(image)
        return {"face": face, "skin": skin, "background": background, "found": True}