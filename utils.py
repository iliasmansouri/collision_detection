from dataclasses import dataclass
import numpy as np


@dataclass
class DataPoints:
    frame_indices: []
    detections: []
    tracks = []

    def __post_init__(self):
        # Determine the length to match (based on frame_indices)
        length = len(self.frame_indices)

        # Fill detections with None if it is None or of a different length
        if not self.detections or len(self.detections) != length:
            self.detections = [None] * length

        # Fill tracks with None if it is None or of a different length
        if not self.tracks or len(self.tracks) != length:
            self.tracks = [None] * length

    def __iter__(self) -> Iterator[np.ndarray]:
        return (
            {"frame_idx": idx, "detections": detections, "tracks": tracks}
            for idx, detections, tracks in zip(
                self.frame_indices, self.detections, self.tracks
            )
        )

    def set_tracks(self, active_tracks):
        self.tracks = active_tracks


@dataclass
class Frame:
    index: int
    cap: None
    img = None

    def __repr__(self) -> str:
        return f"Frame Index: {self.index}"

    def get_img(self):
        if np.shape(self.img) == ():
            self.cap.set(1, self.index)  # Where frame_no is the frame you want
            _, self.img = self.cap.read()  # Read the frame
        return self.img


@dataclass
class Batch:
    frames: list[Frame]

    def __iter__(self) -> Iterator[np.ndarray]:
        return (frame for frame in self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def images(self):
        return [frame.get_img() for frame in self.frames]

    def indices(self):
        return [frame.index for frame in self.frames]
