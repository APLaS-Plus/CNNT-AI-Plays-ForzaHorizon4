# @Author: 蔡俊志
# @Sourceurl: https://github.com/EthanNCai/AI-Plays-ForzaHorizon4
# rebuilt by APLaS
import win32gui
import win32api
import win32con
import win32ui
import numpy as np
import cv2


class ScreenGrabber:
    def __init__(self, display_index=0, region=(0, 0, 1920, 1080)):
        """
        Usage Example
        For frequent call scenarios, create a grabber instance and reuse it
        grabber = ScreenGrabber(display_index=1, region=(0, 0, 1920, 1080))
        while True:
            frame = grabber.grab()
            # process frame...
        """

        # Initialize screen region
        self.display_index = display_index
        self.left, self.top, x2, y2 = region
        self.width = x2 - self.left + 1
        self.height = y2 - self.top + 1

        # Get display monitor offset
        monitors = win32api.EnumDisplayMonitors()
        monitor_info = monitors[self.display_index]
        self.left_offset = monitor_info[2][0]
        self.top_offset = monitor_info[2][1]
        self.left += self.left_offset
        self.top += self.top_offset

        # Initialize DC and bitmap resources
        self.hwin = win32gui.GetDesktopWindow()
        self.hwindc = win32gui.GetWindowDC(self.hwin)
        self.srcdc = win32ui.CreateDCFromHandle(self.hwindc)
        self.memdc = self.srcdc.CreateCompatibleDC()
        self.bmp = win32ui.CreateBitmap()
        self.bmp.CreateCompatibleBitmap(self.srcdc, self.width, self.height)
        self.memdc.SelectObject(self.bmp)

        # Pre-allocate image buffer
        self.img_buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def grab(self):
        """Get screen capture, return image in BGR format"""
        # Copy screen content to bitmap
        self.memdc.BitBlt(
            (0, 0),
            (self.width, self.height),
            self.srcdc,
            (self.left, self.top),
            win32con.SRCCOPY,
        )

        # Get bitmap data
        signedIntsArray = self.bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype="uint8")
        img.shape = (self.height, self.width, 4)

        # Use NumPy slice directly, avoid overhead of cv2.cvtColor
        return img[:, :, :3]  # Keep only BGR channels, discard Alpha channel


    def __del__(self):
        """Release resources"""
        if hasattr(self, "srcdc"):
            self.srcdc.DeleteDC()
        if hasattr(self, "memdc"):
            self.memdc.DeleteDC()
        if hasattr(self, "hwin") and hasattr(self, "hwindc"):
            win32gui.ReleaseDC(self.hwin, self.hwindc)
        if hasattr(self, "bmp"):
            win32gui.DeleteObject(self.bmp.GetHandle())


# Backward compatible function
def grab_screen(display_index=1, region=(0, 0, 1920, 1080)):
    """Wrapper for original function, maintain backward compatibility"""
    grabber = ScreenGrabber(display_index, region)
    img = grabber.grab()
    return img
