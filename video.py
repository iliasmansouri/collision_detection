from dataclasses import dataclass
import cv2
from typing import List

from utils import Frame, Batch


@dataclass
class VideoHandler:
    """
    A class to handle video processing, including creating batches of frames for further analysis.
    Attributes:
        video_path (str): The file path to the video.
        batch_size (int): The number of frames to include in each batch. Default is 16.
        n (int): The step size between frames in a batch. Default is 5.
    Methods:
        __post_init__() -> None: Initializes the video capture and creates batches of frames.
        get_frame_indices() -> List[List[int]]: Generates a list of frame indices to be used for batching.
        create_batches() -> List[Batch]: Creates and returns batches of frames based on the indices.
    """

    video_path: str
    batch_size: int = 16
    n: int = 5

    def __post_init__(self) -> None:
        """
        Initializes the video capture from the given path and creates batches of frames.
        """
        self.cap = cv2.VideoCapture(self.video_path)
        self.batches: List[Batch] = self.create_batches()

    def get_frame_indices(self) -> List[List[int]]:
        """
        Generates a list of frame indices for batching.
        Returns:
            List[List[int]]: A list of lists, where each sublist contains indices of frames for a batch.
        """
        length: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps: int = int(self.cap.get(cv2.CAP_PROP_FPS))
        return [
            list(range(i, min(i + self.batch_size * self.n, length), self.n))
            for i in range(0, length, self.batch_size * self.n)
        ]

    def create_batches(self) -> List[Batch]:
        """
        Creates batches of frames based on the generated indices.
        Returns:
            List[Batch]: A list of Batch objects, each containing a group of frames.
        """
        batches: List[Batch] = []
        for indices in self.get_frame_indices():
            frames: List[Frame] = []
            for idx in indices:
                frames.append(Frame(idx, self.cap))
            batches.append(Batch(frames))
        return batches


if __name__ == "__main__":
    video_path: str = "video.mp4"
    video_handler: VideoHandler = VideoHandler(video_path, batch_size=16, n=15)

    for batch in video_handler.batches:
        print(batch)
