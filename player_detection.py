from ultralytics import YOLO
from dataclasses import dataclass
import cv2
from typing import List, Iterator, Any

from utils import DataPoints
import numpy as np


@dataclass
class Bbx:
    box_xywh: None
    conf: None
    class_id: None

    def __iter__(self) -> Iterator[np.ndarray]:
        return (
            {"box_xywh": box_xywh, "conf": conf, "class_id": class_id}
            for box_xywh, conf, class_id in zip(self.box_xywh, self.conf, self.class_id)
        )


@dataclass
class Model:
    model = YOLO("yolov10x.pt")

    def predict(self, batch):
        results = self.model(batch.images(), classes=[0])
        detections = self.parse_results(results)

        return DataPoints(batch.indices(), detections=detections)

    def parse_results(self, results):
        detections = []
        for result in results:
            boxes = []
            confs = []
            class_ids = []
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0].item()  # Confidence score
                class_id = int(box.cls[0])  # Class ID
                boxes.append(
                    [x1, y1, x2 - x1, y2 - y1]
                )  # Convert to [x, y, w, h] format
                confs.append(confidence)
                class_ids.append(class_id)
            detections.append(Bbx(boxes, confs, class_ids))

        return detections


if __name__ == "__main__":
    image = cv2.imread("image.png")

    m = Model()
    m.predict(image)
    model = YOLO("yolov10x.pt")
    results = model("image.png")

    results[0].show()
