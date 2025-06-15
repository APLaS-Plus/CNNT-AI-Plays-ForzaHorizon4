"""
Computer Vision Image Processing Utilities for Racing Game Analysis.

This module provides specialized image processing functions for racing games,
including color extraction, perspective transformation, and edge detection
optimized for real-time performance.

@Author: Dedsecer
@Sourceurl: https://github.com/DedSecer/AI-Plays-ForzaHorizon4
rebuilt by APLaS
"""

import cv2
import numpy as np


def extract_blue(screen_in):
    """
    Extract blue colored elements from RGB image using HSV color space filtering.

    This function is optimized for extracting blue track markers or UI elements
    in racing games by converting to HSV color space and applying a blue color mask.

    Args:
        screen_in (np.ndarray): Input RGB image array

    Returns:
        np.ndarray: Filtered image with only blue elements visible,
                   rest of the image is masked to black
    """
    # Convert RGB to HSV color space for better color filtering
    # HSV provides better separation of color information from lighting
    hsv = cv2.cvtColor(screen_in, cv2.COLOR_RGB2HSV)

    # Define optimized blue color range in HSV
    # Hue: 90-165 (blue spectrum), Saturation: 38-130, Value: 120-240
    lower_blue = np.array([90, 38, 120])
    upper_blue = np.array([165, 130, 240])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply mask to original image to extract blue regions
    # This preserves the original color information in blue areas
    return cv2.bitwise_and(screen_in, screen_in, mask=mask)


def bird_eye_view(img):
    """
    Transform road image to bird's eye view using perspective transformation.

    This function applies a perspective transformation to convert the driver's
    view to a top-down bird's eye view, which is useful for path planning
    and lane detection in autonomous driving applications.

    The transformation maps a trapezoidal region in the original image
    (representing the road ahead) to a rectangular region in the output.

    Args:
        img (np.ndarray): Input image from driver's perspective

    Returns:
        np.ndarray: Transformed image in bird's eye view perspective
    """
    # Get image dimensions for transformation calculations
    img_size = (img.shape[1], img.shape[0])  # (width, height)

    # Pre-calculate transformation points to avoid repeated computation
    # These ratios are optimized for typical racing game camera angles
    width_half = img.shape[1] * 0.5  # Center horizontal line
    bot_width_half = img.shape[1] * 0.35  # Bottom trapezoid width (70% total / 2)
    mid_width_half = img.shape[1] * 0.025  # Top trapezoid width (5% total / 2)
    height_pos = img.shape[0] * 0.45  # Top trapezoid vertical position
    bottom_pos = img.shape[0] * 0.70  # Bottom trapezoid vertical position

    # Define source points (trapezoid in original image)
    # These points represent the road area in driver's perspective
    src = np.float32(
        [
            [width_half - mid_width_half, height_pos],  # Top-left
            [width_half + mid_width_half, height_pos],  # Top-right
            [width_half + bot_width_half, bottom_pos],  # Bottom-right
            [width_half - bot_width_half, bottom_pos],  # Bottom-left
        ]
    )

    # Define destination points (rectangle in bird's eye view)
    # Maps the trapezoid to a rectangular region with padding
    offset = img_size[0] * 0.25  # 25% padding on each side
    dst = np.float32(
        [
            [offset, 0],  # Top-left
            [img_size[0] - offset, 0],  # Top-right
            [img_size[0] - offset, img_size[1]],  # Bottom-right
            [offset, img_size[1]],  # Bottom-left
        ]
    )

    # Calculate perspective transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Apply perspective transformation with linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped


def bird_view_processing(screen_in, resize_width=160, resize_height=90):
    """
    Complete pipeline for bird's eye view processing with blue extraction.

    This function combines bird's eye view transformation, blue color extraction,
    grayscale conversion, and resizing into a single optimized pipeline for
    real-time racing game analysis.

    Processing Steps:
    1. Transform to bird's eye view perspective
    2. Extract blue colored elements (track markers, UI)
    3. Convert to grayscale for reduced data complexity
    4. Resize to specified dimensions for neural network input

    Args:
        screen_in (np.ndarray): Input RGB image from game screen
        resize_width (int): Target width for output image (default: 160)
        resize_height (int): Target height for output image (default: 90)

    Returns:
        np.ndarray: Processed grayscale image ready for AI model input
    """
    # Step 1: Apply bird's eye view transformation
    processed_image = bird_eye_view(screen_in)

    # Optional: Uncomment to display intermediate result for debugging
    # cv2.imshow('bird_view', cv2.resize(processed_image, (480, 270)))

    # Step 2: Extract blue elements (track markers, navigation aids)
    processed_image = extract_blue(processed_image)

    # Step 3: Convert to grayscale to reduce data complexity and processing time
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Resize to target dimensions for consistent neural network input
    processed_image = cv2.resize(processed_image, (resize_width, resize_height))

    return processed_image


def crop_screen(screen_in, trim_rate=0.3):
    """
    Crop screen edges to remove game UI elements and focus on gameplay area.

    This function removes a specified percentage from the edges of the screen
    to eliminate UI elements, menus, and other non-gameplay visual elements
    that could interfere with AI decision making.

    Args:
        screen_in (np.ndarray): Input screen capture image array
        trim_rate (float): Percentage of screen to trim from edges (0.0-1.0)
                          Default: 0.3 (30% trim)

    Returns:
        np.ndarray: Cropped image with UI elements removed

    Note:
        Currently only trims from top and maintains full width.
        This is optimized for racing games where bottom UI is minimal.
    """
    # Get image dimensions
    height, width, _ = screen_in.shape

    # Calculate padding amounts based on trim rate
    padding_w = int(width * trim_rate)  # Horizontal padding (not currently used)
    padding_h = int(height * trim_rate)  # Vertical padding

    # Crop from top to remove UI elements like minimap, speedometer, etc.
    # Keep full width and crop from top to bottom
    return screen_in[padding_h:, :, :]


def edge_processing(screen_in, resize_width=160, resize_height=90):
    """
    Extract edge features from screen image with Region of Interest (ROI) filtering.

    This function performs edge detection optimized for racing games by:
    1. Resizing image for consistent processing
    2. Applying Canny edge detection with tuned thresholds
    3. Creating a trapezoidal ROI mask to focus on the road ahead
    4. Filtering edges to only include relevant road/track features

    The ROI is shaped like a trapezoid to match the perspective view of roads,
    focusing on the area where the car is likely to drive.

    Args:
        screen_in (np.ndarray): Input RGB screen capture
        resize_width (int): Target width for processing (default: 160)
        resize_height (int): Target height for processing (default: 90)

    Returns:
        np.ndarray: Binary edge image with ROI mask applied,
                   showing only relevant road/track edges
    """
    # Step 1: Resize image for consistent processing and improved performance
    screen_resized = cv2.resize(screen_in, (resize_width, resize_height))

    # Step 2: Apply Canny edge detection with optimized thresholds
    # Lower threshold: 200, Upper threshold: 255
    # These values are tuned for racing game environments with clear track edges
    edges = cv2.Canny(screen_resized, 200, 255)

    # Step 3: Define Region of Interest (ROI) as a trapezoidal area
    # This focuses on the road/track area ahead of the vehicle
    roi_corners = np.array(
        [
            [
                (0, resize_height // 2),  # Left middle point
                (resize_width // 2, 0),  # Top center point
                (resize_width - 1, resize_height // 2),  # Right middle point
                (resize_width - 1, resize_height - 1),  # Bottom right corner
                (0, resize_height - 1),  # Bottom left corner
            ]
        ],
        dtype=np.int32,
    )

    # Step 4: Create and apply ROI mask
    # Initialize mask with zeros (black)
    mask = np.zeros(edges.shape, dtype=np.uint8)
    # Fill ROI polygon with white (255) to create mask
    cv2.fillPoly(mask, roi_corners, 255)

    # Step 5: Apply mask to edges, keeping only edges within ROI
    return cv2.bitwise_and(edges, mask)
