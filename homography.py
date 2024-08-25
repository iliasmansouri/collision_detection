import cv2
import numpy as np
import yaml


def load_points_from_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    image_points = []
    world_points = []

    for point in data["points"]:
        image_points.append(point["pixel"])
        world_points.append(point["world"])

    return np.array(image_points, dtype="float32"), np.array(
        world_points, dtype="float32"
    )


def map_pixel_to_world(pixel_coord, homography_matrix):
    pixel_coord_homogeneous = np.array([pixel_coord[0], pixel_coord[1], 1])
    world_coord_homogeneous = homography_matrix @ pixel_coord_homogeneous
    world_coord = world_coord_homogeneous[:2] / world_coord_homogeneous[2]  # Normalize
    return world_coord


def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    norm_pts = (pts - mean) / std
    return norm_pts, mean, std


if __name__ == "__main__":
    yaml_path = "assets/points.yaml"

    image_points, world_points = load_points_from_yaml(yaml_path)

    homography_matrix, _ = cv2.findHomography(
        image_points, world_points, cv2.RANSAC, ransacReprojThreshold=1.0
    )

    new_pixel_coord = [1608, 263]
    new_world_coord = map_pixel_to_world(new_pixel_coord, homography_matrix)
    print("World coordinates:", new_world_coord)
