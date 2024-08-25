import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from typing import List, Iterator, Any
from dataclasses import dataclass


@dataclass
class Track:
    frame_indices: []
    ids: None
    tracks: None

    def __iter__(self) -> Iterator[np.ndarray]:
        return (
            {"frame_idx": idx, "ids": ids, "tracks": tracks}
            for idx, ids, tracks in zip(self.frame_indices, self.id, self.track)
        )


class Tracker:
    tracker = DeepSort(
        max_age=5,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
    )

    base_model = MobileNetV2(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(128, 128, 3)
    )
    embedder = Model(inputs=base_model.input, outputs=base_model.output)

    def predict(self, batch, datapoints):
        active_tracks = []
        for frame_obj, datapoint in zip(batch, datapoints):
            embeddings = []
            bbx_conf_pair = []
            image = frame_obj.get_img()
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
            for track in tracks:
                if not track.is_confirmed():
                    continue

            active_tracks.append(
                [
                    Track(frame_obj.index, track.track_id, track.to_ltrb())
                    for track in tracks
                ]
            )

        datapoints.tracks = active_tracks
        return datapoints

    def extract_features(self, image, bbox):
        """
        Extract appearance features from a given bounding box using MobileNetV2.
        :param image: The full image
        :param bbox: The bounding box (x1, y1, x2, y2)
        :return: Feature vector (embedding)
        """
        x, y, w, h = bbox
        left = x
        upper = y
        right = x + w
        lower = y + h
        obj_img = image[upper:lower, left:right]
        obj_img = cv2.resize(obj_img, (128, 128))  # Resize to MobileNetV2 input size
        obj_img = preprocess_input(obj_img)  # Preprocess for MobileNetV2
        obj_img = np.expand_dims(obj_img, axis=0)  # Add batch dimension
        features = self.embedder.predict(obj_img)  # Extract features
        return features.flatten()
