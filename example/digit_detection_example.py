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
    # 测试配置
    NUM_TESTS = 100  # 测试次数
    
    # 1. Load model
    model = LiteDigitDetector(input_height=48, input_width=96)

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

    # 2. Load test image (一次性加载)
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

    # 3. 反复测试同一张图片
    pil_loading_times = []
    prediction_times = []
    
    print(f"开始进行 {NUM_TESTS} 次测试...")
    
    for i in range(NUM_TESTS):
        # 测量PIL图片加载时间
        pil_start = time.time()
        pil_image = Image.fromarray(digit_region, mode="L")  # Convert to PIL image in grayscale mode
        pil_end = time.time()
        pil_loading_times.append(pil_end - pil_start)
        
        # 测量预测时间
        predict_start = time.time()
        digits_prediction = predict_with_model(model, pil_image)
        predict_end = time.time()
        prediction_times.append(predict_end - predict_start)
        
        # 显示进度
        if (i + 1) % 20 == 0:
            print(f"已完成 {i + 1}/{NUM_TESTS} 次测试")

    # 计算平均时间
    avg_pil_loading_time = np.mean(pil_loading_times)
    avg_prediction_time = np.mean(prediction_times)
    
    print(f"\n=== 测试结果 (基于 {NUM_TESTS} 次测试) ===")
    print(f"平均PIL图片加载时间: {avg_pil_loading_time:.6f} 秒")
    print(f"平均预测时间: {avg_prediction_time:.6f} 秒")
    print(f"PIL加载时间标准差: {np.std(pil_loading_times):.6f} 秒")
    print(f"预测时间标准差: {np.std(prediction_times):.6f} 秒")
    
    # 显示最后一次预测结果
    print(f"\n最后一次预测结果:")
    print(f"Predicted three digits: {digits_prediction}")
    numeric_value = (
        100 * digits_prediction[0] + 10 * digits_prediction[1] + digits_prediction[2]
    )
    print(f"Predicted combined digit value: {numeric_value}")

    ed = time.time()
    print(f"\n总执行时间: {ed - bg:.3f} 秒")
    
    # Display the captured digit region (只显示一次)
    pil_image = Image.fromarray(digit_region, mode="L")
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(pil_image, cmap="gray")
    plt.title("Captured Digit Region")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
