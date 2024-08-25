from ultralytics import YOLO
from dataclasses import dataclass
import cv2
from typing import List, Iterator, Any, Union

from utils import DataPoints, Batch
import numpy as np


from dataclasses import dataclass
from typing import List, Iterator, Union, Any


@dataclass
class Bbx:
    """
    Represents a bounding box detection with associated confidence and class ID.
    Attributes:
        box_xywh (List[List[int]]): A list of bounding boxes, each defined by [x, y, width, height].
        conf (List[float]): A list of confidence scores for each bounding box.
        class_id (List[int]): A list of class IDs for each bounding box.
    """

    box_xywh: List[List[int]]
    conf: List[float]
    class_id: List[int]

    def __iter__(self) -> Iterator[dict[str, Union[List[int], float]]]:
        """
        Allows iteration over the bounding box, confidence, and class ID data as dictionaries.
        Yields:
            Iterator[dict[str, Union[List[int], float]]]: An iterator where each element is a dictionary
            containing 'box_xywh', 'conf', and 'class_id'.
        """
        return (
            {"box_xywh": box_xywh, "conf": conf, "class_id": class_id}
            for box_xywh, conf, class_id in zip(self.box_xywh, self.conf, self.class_id)
        )


@dataclass
class Model:
    """
    A class that encapsulates a YOLO model for object detection.
    Attributes:
        model (YOLO): An instance of the YOLO model for performing object detection.
    """

    model: YOLO = YOLO("yolov10x.pt")

    def predict(self, batch: "Batch") -> "DataPoints":
        """
        Perform object detection on a batch of images.s
        Args:
            batch (Batch): A batch of images to process.
        Returns:
            DataPoints: The detected data points, including bounding boxes, confidence scores, and class IDs.
        """
        results = self.model(batch.images(), classes=[0])
        detections = self.parse_results(results)
        return DataPoints(batch.indices(), detections=detections)

    def parse_results(self, results: Any) -> List[Bbx]:
        """
        Parse the YOLO model results into a list of Bbx (bounding box) objects.
        Args:
            results (Any): The results from the YOLO model's prediction.
        Returns:
            List[Bbx]: A list of Bbx objects, each containing bounding boxes, confidence scores, and class IDs.
        """
        detections: List[Bbx] = []
        for result in results:
            boxes: List[List[int]] = []
            confs: List[float] = []
            class_ids: List[int] = []
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
