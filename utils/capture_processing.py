import PIL.Image
import cv2
import PIL
import numpy as np
import time
from PIL import ImageGrab


class ImgHelper:
    """
    OpenCV-based screenshot utility class for capturing and processing screen images.
    This class does not contain an initialization function.
    """

    @staticmethod
    def capture_screen():
        """
        Capture screenshot of the entire screen
        :return: screenshot as numpy array
        """
        screenshot = ImageGrab.grab()
        screenshot = np.array(screenshot)
        # Convert from BGR to RGB format
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        return screenshot

    @staticmethod
    def capture_screen_region(x, y, width, height):
        """
        Capture a specific region of the screen
        :param x: x-coordinate of the top-left corner
        :param y: y-coordinate of the top-left corner
        :param width: width of the region
        :param height: height of the region
        :return: screenshot as numpy array
        """
        screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        return screenshot


    @staticmethod
    def capture_region(image, x, y, width, height):
        """
        Crop a specific region from an image using cv2.getRectSubPix
        :param image: input image (numpy array)
        :param x: x-coordinate of the top-left corner
        :param y: y-coordinate of the top-left corner
        :param width: width of the region
        :param height: height of the region
        :return: cropped image region
        """
        # Ensure valid coordinates and dimensions
        img_height, img_width = image.shape[:2]
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        width = max(1, min(width, img_width - x))
        height = max(1, min(height, img_height - y))

        # Calculate center point
        center_x = x + width / 2
        center_y = y + height / 2

        # Get sub-image using cv2.getRectSubPix
        return cv2.getRectSubPix(image, (int(width), int(height)), (center_x, center_y))

    @staticmethod
    def save_screenshot(image, filename):
        """
        Save screenshot to file
        :param image: screenshot as numpy array
        :param filename: filename to save to
        """
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    @staticmethod
    def convert_to_grayscale(image):
        """
        Convert image to grayscale
        :param image: color image
        :return: grayscale image
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    

    @staticmethod
    def draw_rectangle(image, x, y, width, height, color=(0, 255, 0), thickness=2):
        """
        Draw a rectangle on the image
        :param image: image to draw on
        :param x: x-coordinate of the top-left corner
        :param y: y-coordinate of the top-left corner
        :param width: width of the rectangle
        :param height: height of the rectangle
        :param color: color (default is green)
        :param thickness: line thickness
        :return: image with rectangle drawn
        """
        result = image.copy()
        cv2.rectangle(result, (x, y), (x + width, y + height), color, thickness)
        return result

    @staticmethod
    def show_image(image):
        image = PIL.Image.fromarray(image)
        image.show()

    @staticmethod
    def warp_perspective(image, points, target_width, target_height):
        """
        Stretch any quadrilateral image region into a rectangle of specified size

        :param image: input image
        :param points: four corner points of the quadrilateral in the original image,
                      format [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                      points should be ordered: top-left, top-right, bottom-right, bottom-left
        :param target_width: width of the target rectangle
        :param target_height: height of the target rectangle
        :return: warped rectangular image
        """
        # Ensure points are in order: top-left, top-right, bottom-right, bottom-left
        src = np.float32(points)

        # Define the four corner points of the target rectangle
        dst = np.float32(
            [
                [0, 0],  # top-left
                [target_width - 1, 0],  # top-right
                [target_width - 1, target_height - 1],  # bottom-right
                [0, target_height - 1],  # bottom-left
            ]
        )

        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src, dst)

        # Apply perspective transformation
        result = cv2.warpPerspective(image, matrix, (target_width, target_height))

        return result

    def capture_digit_region(img):
        """
        Capture the digit region from the image.
        The region is defined as a rectangle with fixed coordinates.
        """
        # Define the coordinates for the digit region (x, y, width, height)
        x, y, w, h = 1710, 912, 160, 96
        # Crop the image to get the digit region
        digit_region = img[y:y+h, x:x+w]
        return digit_region
