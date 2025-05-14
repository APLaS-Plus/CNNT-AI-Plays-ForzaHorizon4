import cv2
import pathlib
from PIL import Image
import sys
import matplotlib.pyplot as plt

ROOT_DIR = pathlib.Path(__file__).resolve().parent

UTILS_DIR = ROOT_DIR / ".."
UTILS_DIR = str(UTILS_DIR.resolve())

sys.path.append(UTILS_DIR)

from utils.vendor.cv_img_processing import (
    bird_view_processing,
    bird_eye_view,
    extract_blue,
)

TEST_PNG = ROOT_DIR / ".." / "testpng" / "night.jpg"

# Load test image
img = cv2.imread(str(TEST_PNG), cv2.IMREAD_COLOR_RGB)

# Apply bird eye view transformation
bird_view = bird_eye_view(img)

# Extract blue features from the bird eye view
blue_img = extract_blue(bird_view)

# Create a figure with 3 subplots to visualize the transformation process
plt.figure(figsize=(15, 5))

# Plot original image
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

# Plot bird eye view
plt.subplot(1, 3, 2)
plt.imshow(bird_view)
plt.title("Bird's Eye View")
plt.axis("off")

# Plot blue features extraction
plt.subplot(1, 3, 3)
plt.imshow(blue_img, cmap="gray")
plt.title("Blue Features Extracted")
plt.axis("off")

plt.suptitle("Image Processing Pipeline", fontsize=16)
plt.tight_layout()

# Uncomment to display the figure instead of saving
plt.show()
