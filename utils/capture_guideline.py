from .capture_processing import ImgHelper
from .vendor import cv_img_processing as cv_img_processing
from .vendor import grabscreen
import cv2


class CaptureGuideline:
    def __init__(self):
        self.grabber = grabscreen.ScreenGrabber()

    def get_digit(self, frame):
        return ImgHelper.capture_digit_region(frame)

    def get_bule_bird_eye_view(self, frame):
        bird_eye_view = cv_img_processing.bird_eye_view(frame)
        resized_bird_eye_view = cv2.resize(bird_eye_view, (240, 136))
        blue_img = cv_img_processing.extract_blue(
            cv2.cvtColor(resized_bird_eye_view, cv2.COLOR_BGR2RGB)
        )
        return blue_img

    def get_currunt_key_region(self):
        frame = self.grabber.grab()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        digit_region = self.get_digit(frame)
        blue_bird_eye_view = self.get_bule_bird_eye_view(frame)
        return digit_region, blue_bird_eye_view

    def get_frame(self):
        frame = self.grabber.grab()
        return frame
