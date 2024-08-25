from dataclasses import dataclass
import cv2

from utils import Frame, Batch


@dataclass
class VideoHandler:
    video_path: str
    batch_size: int = 16
    n: int = 5
    # target_size: tuple = (640, 480)

    def __post_init__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.batches = self.create_batches()
        # self.frame_grabber = self.FrameGrabber(
        #     self.video_path, self.frame_indices, self.target_size
        # )

    def get_frame_indices(self):
        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        return [
            list(range(i, min(i + self.batch_size * self.n, length), self.n))
            for i in range(0, length, self.batch_size * self.n)
        ]

    def create_batches(self):
        batches = []
        for indices in self.get_frame_indices():
            frames = []
            for idx in indices:
                frames.append(Frame(idx, self.cap))
            batches.append(Batch(frames))
        return batches


if __name__ == "__main__":
    video_path = "video.mp4"
    video_handler = VideoHandler(video_path, batch_size=16, n=15)

    for batch in video_handler.batches:
        print(batch)
