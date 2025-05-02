import cv2
import pathlib
from PIL import Image
import sys

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

img = cv2.imread(str(TEST_PNG), cv2.IMREAD_COLOR_RGB)
showimg = Image.fromarray(img)
showimg.show()

# enhance_img = enhance_image(img)
# showimg = Image.fromarray(enhance_img)

# cropped_screen = crop_screen(img)
bird_view = bird_eye_view(img)

showimg = Image.fromarray(bird_view)
showimg.show()

bule_img = extract_blue(bird_view)
# resized_image = bird_view_processing(img)

showimg = Image.fromarray(bule_img)
showimg.show()