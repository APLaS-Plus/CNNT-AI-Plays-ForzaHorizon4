"""
High-Performance Screen Capture Utility for Real-Time Applications.

This module provides optimized screen capture functionality using Windows API
for real-time game monitoring and AI training data collection. It's specifically
designed for high-frequency screen grabbing with minimal performance overhead.

@Author: 蔡俊志
@Sourceurl: https://github.com/EthanNCai/AI-Plays-ForzaHorizon4
rebuilt by APLaS
"""

import win32gui
import win32api
import win32con
import win32ui
import numpy as np
import cv2


class ScreenGrabber:
    """
    High-performance screen capture class optimized for real-time applications.

    This class provides efficient screen capture by pre-allocating resources
    and maintaining persistent Windows GDI objects to minimize overhead
    between capture calls. It's designed for applications requiring frequent
    screen captures, such as AI training data collection or real-time monitoring.

    Features:
    - Multi-monitor support with display index selection
    - Configurable capture region for performance optimization
    - Resource pre-allocation for minimal per-frame overhead
    - Automatic resource cleanup to prevent memory leaks

    Attributes:
        display_index (int): Index of the display monitor to capture from
        left, top (int): Top-left coordinates of capture region
        width, height (int): Dimensions of capture region
        hwin, hwindc, srcdc, memdc: Windows GDI handles for screen capture
        bmp: Windows bitmap object for image data storage
        img_buffer (np.ndarray): Pre-allocated numpy array for image data
    """

    def __init__(self, display_index=0, region=(0, 0, 1920, 1080)):
        """
        Initialize ScreenGrabber with specified display and capture region.

        This constructor sets up all necessary Windows GDI resources for
        efficient screen capture. Resources are pre-allocated to minimize
        overhead during actual capture operations.

        Args:
            display_index (int): Index of display monitor to capture from
                               0 = primary monitor, 1 = secondary monitor, etc.
            region (tuple): Capture region as (left, top, right, bottom)
                          coordinates in pixels

        Example:
            # Capture full primary monitor
            grabber = ScreenGrabber(0, (0, 0, 1920, 1080))

            # Capture game window area on secondary monitor
            grabber = ScreenGrabber(1, (100, 100, 1820, 980))
        """
        # Store display configuration
        self.display_index = display_index
        self.left, self.top, x2, y2 = region
        self.width = x2 - self.left + 1
        self.height = y2 - self.top + 1

        # Get display monitor information and calculate offset coordinates
        # This handles multi-monitor setups where monitors may have different origins
        monitors = win32api.EnumDisplayMonitors()
        monitor_info = monitors[self.display_index]
        self.left_offset = monitor_info[2][0]  # Monitor's left coordinate
        self.top_offset = monitor_info[2][1]  # Monitor's top coordinate

        # Adjust capture coordinates to account for monitor offset
        self.left += self.left_offset
        self.top += self.top_offset

        # Initialize Windows GDI resources for screen capture
        # These objects are reused across multiple captures for efficiency

        # Get desktop window handle and device context
        self.hwin = win32gui.GetDesktopWindow()
        self.hwindc = win32gui.GetWindowDC(self.hwin)

        # Create source device context from desktop DC
        self.srcdc = win32ui.CreateDCFromHandle(self.hwindc)

        # Create memory device context for bitmap operations
        self.memdc = self.srcdc.CreateCompatibleDC()

        # Create bitmap object compatible with source DC
        self.bmp = win32ui.CreateBitmap()
        self.bmp.CreateCompatibleBitmap(self.srcdc, self.width, self.height)

        # Select bitmap into memory DC for drawing operations
        self.memdc.SelectObject(self.bmp)

        # Pre-allocate numpy array buffer for image data
        # BGRA format: Blue, Green, Red, Alpha channels
        self.img_buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def grab(self):
        """
        Capture screen region and return as BGR image array.

        This method performs the actual screen capture using Windows BitBlt
        operation, which is highly optimized for copying pixel data between
        device contexts. The captured data is converted to a numpy array
        in BGR format suitable for OpenCV operations.

        Returns:
            np.ndarray: Captured screen image in BGR format with shape
                       (height, width, 3). Alpha channel is discarded
                       for compatibility with standard image processing.

        Performance Notes:
        - BitBlt operation is hardware-accelerated when possible
        - Memory allocation is minimized through pre-allocated buffers
        - Color channel ordering is optimized for OpenCV compatibility
        """
        # Perform screen capture using BitBlt (Block Transfer)
        # This copies pixels from source DC (screen) to memory DC (bitmap)
        self.memdc.BitBlt(
            (0, 0),  # Destination coordinates in bitmap
            (self.width, self.height),  # Dimensions to copy
            self.srcdc,  # Source device context (screen)
            (self.left, self.top),  # Source coordinates on screen
            win32con.SRCCOPY,  # Copy operation mode
        )

        # Extract bitmap data as raw bytes
        signedIntsArray = self.bmp.GetBitmapBits(True)

        # Convert raw bytes to numpy array
        img = np.frombuffer(signedIntsArray, dtype="uint8")

        # Reshape flat array to image dimensions with BGRA channels
        img.shape = (self.height, self.width, 4)

        # Return BGR channels only (discard Alpha channel)
        # This provides compatibility with OpenCV which expects BGR format
        return img[:, :, :3]  # Keep only BGR channels, discard Alpha channel

    def __del__(self):
        """
        Destructor to properly release Windows GDI resources.

        This method ensures all allocated Windows resources are properly
        released when the ScreenGrabber object is destroyed. Proper cleanup
        prevents resource leaks and system handle exhaustion.

        Resources cleaned up:
        - Device contexts (DC)
        - Bitmap objects
        - Window handles

        Note: This method is called automatically by Python's garbage collector
        """
        # Release device contexts in reverse order of creation
        if hasattr(self, "srcdc"):
            self.srcdc.DeleteDC()
        if hasattr(self, "memdc"):
            self.memdc.DeleteDC()

        # Release window device context
        if hasattr(self, "hwin") and hasattr(self, "hwindc"):
            win32gui.ReleaseDC(self.hwin, self.hwindc)

        # Delete bitmap object
        if hasattr(self, "bmp"):
            win32gui.DeleteObject(self.bmp.GetHandle())


def grab_screen(display_index=1, region=(0, 0, 1920, 1080)):
    """
    Legacy wrapper function for backward compatibility with existing code.

    This function provides a simple interface for one-time screen captures
    without the need to manage a ScreenGrabber instance. However, for
    applications requiring frequent captures, using ScreenGrabber class
    directly is more efficient.

    Args:
        display_index (int): Index of display monitor (default: 1 for secondary)
        region (tuple): Capture region as (left, top, right, bottom)
                       Default: (0, 0, 1920, 1080) for full HD capture

    Returns:
        np.ndarray: Captured screen image in BGR format

    Usage Example:
        # One-time capture
        img = grab_screen(1, (0, 0, 1920, 1080))

        # For frequent captures, use ScreenGrabber class instead:
        grabber = ScreenGrabber(1, (0, 0, 1920, 1080))
        while True:
            img = grabber.grab()
            # process image...
    """
    # Create temporary ScreenGrabber instance
    grabber = ScreenGrabber(display_index, region)

    # Perform single capture and return result
    img = grabber.grab()

    # ScreenGrabber destructor will automatically clean up resources
    return img
