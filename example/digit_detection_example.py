import torch
import numpy as np
import cv2
import time
import os
import pathlib
import sys
from PIL import Image
import matplotlib.pyplot as plt

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
UTILS_DIR = ROOT_DIR / ".."
UTILS_DIR = str(UTILS_DIR.resolve())

# print("Root directory:", ROOT_DIR)
# print("Utils directory:", UTILS_DIR)

sys.path.append(UTILS_DIR)  # Add utils directory to path
from utils.model.lite_digit_detector import LiteDigitDetector
from utils.capture_processing import ImgHelper


def predict_with_model(model, image):
    """Use model to predict digits in image"""
    digits = model.predict(image)
    return digits


def main():
    bg = time.time()
    # 1. Load model
    model = LiteDigitDetector(input_height=48, input_width=80)

    # Load pretrained weights
    model_path = ROOT_DIR / ".." / "model" / "LDD" / "best_digit_model.pth"
    try:
        model.load_weights(str(model_path))
        print("Successfully loaded model weights")
    except:
        print("Pretrained model weights not found, using randomly initialized model")

    # Set model to evaluation mode
    model.eval()

    load_model = time.time()
    print("Model loading time:", load_model - bg)

    # 2. Load test image
    image_path = (
        ROOT_DIR / ".." / "testpng" / "night.jpg"
    )  # Modify to your test image path
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    digit_region = ImgHelper.capture_digit_region(
        img
    )  # Crop the digit region from the image

    if digit_region is None:
        print("Error: Could not capture digit region")
        return

    # Apply image preprocessing steps
    # Apply median filter to remove noise
    median = cv2.medianBlur(digit_region, 3)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(median)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
    )

    # Morphological operations to enhance character contours
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Convert processed image to PIL format
    pil_image = Image.fromarray(morph)

    load_pil_image = time.time()
    print("PIL image loading time:", load_pil_image - load_model)

    # 3. Make prediction
    bg_predict = time.time()
    digits_prediction = predict_with_model(model, pil_image)
    predict_img = time.time()
    print("Prediction time:", predict_img - bg_predict)
    print(f"Predicted three digits: {digits_prediction}")
    print(
        f"Combined digit value: {digits_prediction[0] * 100 + digits_prediction[1] * 10 + digits_prediction[2]}"
    )

    # Calculate the value of the three-digit number
    numeric_value = (
        100 * digits_prediction[0] + 10 * digits_prediction[1] + digits_prediction[2]
    )
    print(f"Predicted combined digit value: {numeric_value}")

    ed = time.time()
    print("Total execution time:", ed - bg)
    # Display the captured digit region
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(digit_region, cmap="gray")
    plt.title("Captured Digit Region")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
