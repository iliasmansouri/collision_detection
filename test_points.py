import cv2
import yaml


def load_points_from_yaml(yaml_path):
    """Load points from a YAML file."""
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return data["points"]


def annotate_image(image_path, points):
    """Annotate the image with pixel coordinates and world coordinates."""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # Define colors
    bullet_color = (0, 0, 255)  # Red color for bullet
    text_color = (255, 255, 255)  # White color for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    for point in points:
        pixel = point["pixel"]
        world = point["world"]

        # Draw a circle at the pixel coordinates (bullet)
        cv2.circle(img, tuple(pixel), 5, bullet_color, -1)

        # Draw the world coordinates as text
        text = f"({world[0]}, {world[1]})"
        text_position = (pixel[0] + 10, pixel[1] - 10)  # Adjust text position
        cv2.putText(
            img, text, text_position, font, font_scale, text_color, font_thickness
        )

    # Display the image
    cv2.namedWindow("Annotated Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Annotated Image", img)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace these paths with your actual file paths
    yaml_path = "assets/points.yaml"
    image_path = "assets/image.png"

    # Load points from YAML
    points = load_points_from_yaml(yaml_path)

    # Annotate and display the image
    annotate_image(image_path, points)
