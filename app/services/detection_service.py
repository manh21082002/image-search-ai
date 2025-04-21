from ultralytics import YOLO
from PIL import Image
import numpy as np
from typing import List, Tuple

class YoloV8Detector:
    def __init__(self, model_path: str = "models/yolov8_food.pt"):
        self.model = YOLO(model_path)  # model YOLOv8n đã fine-tune
        self.model.fuse()  # tăng tốc độ inference

    def detect_objects(self, image: Image.Image, conf_thresh: float = 0.1) -> List[Tuple[int, int, int, int]]:
        # Chuyển ảnh PIL → numpy array
        img_array = np.array(image)

        # Dự đoán không hiển thị log
        results = self.model.predict(source=img_array, conf=conf_thresh, verbose=False)

        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((int(x1), int(y1), int(x2), int(y2)))

        return boxes
