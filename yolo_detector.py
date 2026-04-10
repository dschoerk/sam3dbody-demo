"""
Lightweight YOLO-based person detector that matches the HumanDetector interface
from sam-3d-body/tools/build_detector.py.

Kept in the main repo so the vendored sam-3d-body tree stays unmodified.
"""

import numpy as np


class YoloDetector:
    def __init__(self, model_name: str = "yolo11n.pt", device: str = "cuda"):
        from ultralytics import YOLO
        self.model = YOLO(model_name).to(device)

    def run_human_detection(
        self,
        img,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        **kwargs,
    ) -> np.ndarray:
        """Return (N, 4) xyxy bounding boxes for detected persons. img is BGR uint8."""
        results = self.model(img, classes=[0], conf=bbox_thr, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            h, w = img.shape[:2]
            return np.array([[0, 0, w, h]], dtype=np.float32)
        sorted_idx = np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]))
        return boxes[sorted_idx]
