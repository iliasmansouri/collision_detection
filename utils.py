from dataclasses import dataclass
from typing import Iterator, List, Optional, Any, Dict, Union
import numpy as np


@dataclass
class DataPoints:
    """
    A class to store and manage data points for frames, including detections and tracking information.
    Attributes:
        frame_indices (List[int]): A list of indices corresponding to the frames.
        detections (List[Optional[Any]]): A list of detection data for each frame.
        tracks (List[Optional[Any]]): A list of tracking data for each frame. Initialized to None by default.
    Methods:
        __post_init__(): Ensures that the detections and tracks lists are initialized correctly.
        __iter__(): Allows iteration over the data points as dictionaries containing frame indices, detections, and tracks.
        set_tracks(active_tracks: List[Optional[Any]]) -> None: Sets the tracking data for the frames.
    """

    frame_indices: List[int]
    detections: List[Optional[Any]]
    tracks: List[Optional[Any]] = None

    def __post_init__(self) -> None:
        """
        Ensures that the detections and tracks lists are of the correct length.
        If they are not, they are filled with None to match the length of frame_indices.
        """
        # Determine the length to match (based on frame_indices)
        length: int = len(self.frame_indices)

        # Fill detections with None if it is None or of a different length
        if not self.detections or len(self.detections) != length:
            self.detections = [None] * length

        # Fill tracks with None if it is None or of a different length
        if not self.tracks or len(self.tracks) != length:
            self.tracks = [None] * length

    def __iter__(self) -> Iterator[Dict[str, Union[int, Optional[Any]]]]:
        """
        Allows iteration over the data points, yielding dictionaries with frame index, detections, and tracks.
        Yields:
            Iterator[Dict[str, Union[int, Optional[Any]]]]: An iterator over the data points.
        """
        return (
            {"frame_idx": idx, "detections": detections, "tracks": tracks}
            for idx, detections, tracks in zip(
                self.frame_indices, self.detections, self.tracks
            )
        )

    def set_tracks(self, active_tracks: List[Optional[Any]]) -> None:
        """
        Sets the tracking data for the frames.
        Args:
            active_tracks (List[Optional[Any]]): A list of tracking data to be assigned to the frames.
        """
        self.tracks = active_tracks


@dataclass
class Frame:
    """
    A class representing a single video frame.
    Attributes:
        index (int): The index of the frame within the video.
        cap (Any): A video capture object (e.g., cv2.VideoCapture) used to retrieve the frame image.
        img (Optional[np.ndarray]): The image data of the frame. Initialized to None by default.
    Methods:
        __repr__() -> str: Returns a string representation of the frame object.
        get_img() -> np.ndarray: Retrieves the image data of the frame, loading it from the video capture object if necessary.
    """

    index: int
    cap: Any  # Assuming cap is a cv2.VideoCapture or similar object
    img: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        """
        Returns a string representation of the frame object.
        Returns:
            str: A string representation indicating the frame index.
        """
        return f"Frame Index: {self.index}"

    def get_img(self) -> np.ndarray:
        """
        Retrieves the image data of the frame. If the image data is not already loaded, it is read from the video capture object.
        Returns:
            np.ndarray: The image data of the frame.
        """
        if self.img is None or np.shape(self.img) == ():
            self.cap.set(1, self.index)  # Where frame_no is the frame you want
            _, self.img = self.cap.read()  # Read the frame
        return self.img

    def clear(self):
        self.img = None


@dataclass
class Batch:
    """
    A class representing a batch of video frames.
    Attributes:
        frames (List[Frame]): A list of Frame objects in the batch.
    Methods:
        __iter__() -> Iterator[Frame]: Allows iteration over the frames in the batch.
        __len__() -> int: Returns the number of frames in the batch.
        images() -> List[np.ndarray]: Retrieves the images of all frames in the batch.
        indices() -> List[int]: Retrieves the indices of all frames in the batch.
    """

    frames: List[Frame]

    def __iter__(self) -> Iterator[Frame]:
        """
        Allows iteration over the frames in the batch.
        Returns:
            Iterator[Frame]: An iterator over the Frame objects in the batch.
        """
        return iter(self.frames)

    def __len__(self) -> int:
        """
        Returns the number of frames in the batch.
        Returns:
            int: The number of frames.
        """
        return len(self.frames)

    def images(self) -> List[np.ndarray]:
        """
        Retrieves the images of all frames in the batch.
        Returns:
            List[np.ndarray]: A list of image data for each frame in the batch.
        """
        return [frame.get_img() for frame in self.frames]

    def indices(self) -> List[int]:
        """
        Retrieves the indices of all frames in the batch.
        Returns:
            List[int]: A list of indices for each frame in the batch.
        """
        return [frame.index for frame in self.frames]

    def clear_images(self) -> None:
        [frame.clear() for frame in self.frames]
