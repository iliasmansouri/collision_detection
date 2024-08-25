from video import VideoHandler
from player_detection import Model
from player_tracking import Tracker
from collision_detection import Detector
import cv2
import argparse


def visualize(batch, datapoints):
    for frame_obj, tracks in zip(batch, datapoints.tracks):
        # Draw the bounding boxes and track IDs on the frame
        for track in tracks:
            # print(track)
            track_id = track.ids
            ltrb = track.tracks  # Get the left, top, right, bottom bounding box

            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame_obj.get_img(), (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame_obj.get_img(),
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )

        # Display the frame with drawn bounding boxes and track information
        cv2.imshow("Frame", frame_obj.get_img())
        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video processing script")

    parser.add_argument(
        "--video_path", type=str, default="video.mp4", help="Path to the video file"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Frame Batch size")
    parser.add_argument(
        "--n", type=int, default=5, help="Take every n-th frame for processing"
    )

    args = parser.parse_args()

    video_handler = VideoHandler(args.video_path, batch_size=args.batch_size, n=args.n)
    detector = Model()
    tracker = Tracker()
    collision_detector = Detector()

    for batch in video_handler.batches:
        datapoints = detector.predict(batch)
        datapoints = tracker.predict(batch, datapoints)

        collision = collision_detector.check_for_collisions(datapoints.tracks)
        if collision:
            visualize(batch, datapoints)
        batch.clear_images()
