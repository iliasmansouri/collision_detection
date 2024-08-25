from typing import List, Tuple, Dict


class Detector:
    def __init__(self, min_overlap_ratio: float = 0.5) -> None:
        self.min_overlap_ratio = min_overlap_ratio

    def check_for_collisions(
        self, active_tracks: List[List["Track"]]
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Check for collisions between humans based on their bounding boxes.
        :param active_tracks: List of lists of Track objects, where each list corresponds to tracks in a frame.
        :return: List of colliding pairs of bounding boxes.
        """
        collisions: List[Tuple[List[int], List[int]]] = []

        # Iterate through each frame's tracks
        for frame_tracks in active_tracks:
            # Create a dictionary to map track IDs to bounding boxes
            track_dict: Dict[int, List[int]] = {
                track.ids: track.tracks for track in frame_tracks
            }

            track_ids: List[int] = list(track_dict.keys())
            num_tracks: int = len(track_ids)

            # Check for collisions within the current frame
            for i in range(num_tracks):
                for j in range(i + 1, num_tracks):
                    track_id1: int = track_ids[i]
                    track_id2: int = track_ids[j]

                    bbox1: List[int] = track_dict[track_id1]
                    bbox2: List[int] = track_dict[track_id2]

                    if self.is_collision(bbox1, bbox2):
                        print(
                            f"Collision detected between track {track_id1} and track {track_id2}!"
                        )
                        collisions.append((bbox1, bbox2))

        return collisions

    def is_collision(self, bbox1: List[int], bbox2: List[int]) -> bool:
        """
        Determine if two bounding boxes are colliding based on a minimum overlap ratio.
        :param bbox1: The first bounding box (x1, y1, x2, y2).
        :param bbox2: The second bounding box (x1, y1, x2, y2).
        :return: Boolean indicating whether a collision occurs.
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate the dimensions of each bounding box
        width1: int = x2_1 - x1_1
        height1: int = y2_1 - y1_1
        width2: int = x2_2 - x1_2
        height2: int = y2_2 - y1_2

        # Calculate the overlap in both x and y dimensions
        overlap_x: int = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        overlap_y: int = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))

        # Calculate the overlap ratios for both dimensions
        overlap_ratio_x: float = overlap_x / min(width1, width2)
        overlap_ratio_y: float = overlap_y / min(height1, height2)

        # Check if both overlap ratios meet the minimum required overlap ratio
        if (
            overlap_ratio_x >= self.min_overlap_ratio
            and overlap_ratio_y >= self.min_overlap_ratio
        ):
            return True
        else:
            return False  # Not enough overlap, no collision
