import pathlib
import cv2
import time
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent

UTILS_DIR = ROOT_DIR / ".."
UTILS_DIR = str(UTILS_DIR.resolve())

sys.path.append(UTILS_DIR)
from utils.capture_guideline import CaptureGuideline

cgl = CaptureGuideline()

while True:
    bg = time.time()
    digit_region, blue_bird_eye_view = cgl.get_currunt_key_region()
    cv2.imshow("digit_region", digit_region)
    cv2.imshow("blue_bird_eye_view", blue_bird_eye_view)
    # frame = cgl.get_frame()
    # cv2.imshow('frame', frame)
    getframe = time.time()
    print(f"FPS: {1 / (getframe - bg)}")
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
