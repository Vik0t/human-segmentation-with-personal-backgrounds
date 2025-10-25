import mediapipe as mp
import cv2
import numpy as np

class RealTimeSegmentation:
    def __init__(self, use_temporal_filter=True, temporal_alpha=0.8):
        self.use_temporal_filter = use_temporal_filter
        self.temporal_alpha = temporal_alpha
        self.prev_mask = None

        # Используем MediaPipe Selfie Segmentation
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.model = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def infer(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.process(rgb_frame)
        mask = results.segmentation_mask
        mask = (mask > 0.5).astype(np.uint8) * 255

        # Временная фильтрация
        if self.use_temporal_filter and self.prev_mask is not None:
            mask = cv2.addWeighted(mask, self.temporal_alpha, self.prev_mask, 1 - self.temporal_alpha, 0)
        self.prev_mask = mask

        # Морфологические операции
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask, frame