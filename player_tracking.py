import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from typing import List, Iterator, Any, Dict, Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from typing import List, Iterator, Any, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Track:
    """
    A class representing a tracking record for objects across video frames.
    Attributes:
        frame_indices (List[int]): A list of frame indices where the object is tracked.
        ids (int): The ID associated with the tracked object.
        tracks (np.ndarray): An array containing the bounding box coordinates of the tracked object.
    Methods:
        __iter__() -> Iterator[Dict[str, Any]]: Allows iteration over the tracking data as dictionaries.
    """

    frame_indices: List[int]
    ids: int
    tracks: np.ndarray

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Allows iteration over the tracking data, yielding dictionaries with frame index, IDs, and track information.
        Yields:
            Iterator[Dict[str, Any]]: An iterator over the tracking data.
        """
        return (
            {"frame_idx": idx, "ids": ids, "tracks": tracks}
            for idx, ids, tracks in zip(self.frame_indices, self.ids, self.tracks)
        )


class Tracker:
    """
    A class for tracking objects in video frames using DeepSort and MobileNetV2 for feature extraction.
    Attributes:
        tracker (DeepSort): An instance of the DeepSort tracker for managing object tracking.
        base_model (Model): The base MobileNetV2 model used for feature extraction, without the top layers.
        embedder (Model): The feature extraction model that outputs embeddings based on MobileNetV2.
    Methods:
        predict(batch: Any, datapoints: Any) -> Any: Predicts tracks for objects in a batch of frames.
        extract_features(image: np.ndarray, bbox: List[int]) -> np.ndarray: Extracts features from a bounding box within an image.
    """

    tracker: DeepSort = DeepSort(
        max_age=5,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
    )

    base_model: Model = MobileNetV2(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(128, 128, 3)
    )
    embedder: Model = Model(inputs=base_model.input, outputs=base_model.output)

    def predict(self, batch: Any, datapoints: Any) -> Any:
        """
        Predicts and updates tracks for objects detected in a batch of frames.
        Args:
            batch (Any): A batch of frames to process.
            datapoints (Any): Corresponding data points that include detection information.
        Returns:
            Any: Updated datapoints with track information included.
        """
        active_tracks: List[List[Track]] = []
        for frame_obj, datapoint in zip(batch, datapoints):
            embeddings: List[np.ndarray] = []
            bbx_conf_pair: List[Tuple[List[int], float]] = []
            image: np.ndarray = frame_obj.get_img()
            for bbx_obj in datapoint["detections"]:
                embeddings.append(self.extract_features(image, bbx_obj["box_xywh"]))
                bbx_conf_pair.append(
                    (
                        bbx_obj["box_xywh"],
                        bbx_obj["conf"],
                    )
                )

            tracks = self.tracker.update_tracks(
                bbx_conf_pair, frame=image, embeds=embeddings
            )
            confirmed_tracks: List[Track] = [
                Track(frame_obj.index, track.track_id, track.to_ltrb())
                for track in tracks
                if track.is_confirmed()
            ]

            active_tracks.append(confirmed_tracks)

        datapoints.tracks = active_tracks
        return datapoints

    def extract_features(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extracts appearance features from a given bounding box within an image using MobileNetV2.
        Args:
            image (np.ndarray): The full image from which features are to be extracted.
            bbox (List[int]): The bounding box coordinates (x, y, w, h).
        Returns:
            np.ndarray: A flattened feature vector (embedding) representing the object within the bounding box.
        """
        x, y, w, h = bbox
        left: int = x
        upper: int = y
        right: int = x + w
        lower: int = y + h
        obj_img: np.ndarray = image[upper:lower, left:right]
        obj_img = cv2.resize(obj_img, (128, 128))  # Resize to MobileNetV2 input size
        obj_img = preprocess_input(obj_img)  # Preprocess for MobileNetV2
        obj_img = np.expand_dims(obj_img, axis=0)  # Add batch dimension
        features: np.ndarray = self.embedder.predict(obj_img)  # Extract features
        return features.flatten()
