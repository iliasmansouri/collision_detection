import cv2
import numpy as np

# Load the fisheye image
img = cv2.imread("image.png")
h, w = img.shape[:2]

# Initialize the mapping for undistortion
map_x = np.zeros((h, w), dtype=np.float32)
map_y = np.zeros((h, w), dtype=np.float32)

# Approximate parameters for fisheye correction
# Adjust the coefficient for best results
# a=0,1
a = 0.04  # Radial distortion factor (adjust as needed)

# Create the undistortion mapping
for i in range(h):
    for j in range(w):
        # Normalize coordinates to [-1, 1]
        x = (j - w / 2) / (w / 2)
        y = (i - h / 2) / (h / 2)
        r = np.sqrt(x**2 + y**2)

        # Apply radial distortion correction
        if r != 0:
            x_corrected = x / (1 + a * r**2)
            y_corrected = y / (1 + a * r**2)
        else:
            x_corrected = x
            y_corrected = y

        # Map back to pixel coordinates
        map_x[i, j] = x_corrected * (w / 2) + w / 2
        map_y[i, j] = y_corrected * (h / 2) + h / 2

# Apply the remapping to correct the fisheye effect
undistorted_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

# Save and display the corrected image
cv2.imwrite("undistorted_image_approx.jpg", undistorted_img)
cv2.imshow("Undistorted Image", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
