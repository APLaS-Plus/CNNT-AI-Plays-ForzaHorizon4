import sys
import pathlib
import time
import shutil
import cv2
import os
import numpy as np
from PIL import Image

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

UTILS_DIR = ROOT_DIR
UTILS_DIR = str(UTILS_DIR.resolve())

sys.path.append(UTILS_DIR)
from utils.capture_guideline import CaptureGuideline
from utils.input_monitor import InputMonitor, KeyboardInterrupt
from utils.model.lite_digit_detector import LiteDigitDetector

RAW_DATA_DIR = ROOT_DIR / "raw_data" / "CNNT"


def augment_dataset(data_dir):
    """
    Augment the dataset by horizontal flipping the images and inverting the steering values

    Args:
        data_dir: Original dataset directory
    """
    print(f"Starting dataset augmentation with horizontal flipping...")

    # Create directory for augmented dataset
    augmented_data_dir = data_dir.parent / f"{data_dir.name}_augmented"
    if not augmented_data_dir.exists():
        augmented_data_dir.mkdir(parents=True, exist_ok=True)

    # Create train and validation subdirectories for augmented data
    aug_train_path = augmented_data_dir / "train"
    aug_val_path = augmented_data_dir / "val"
    aug_train_path.mkdir(parents=True, exist_ok=True)
    aug_val_path.mkdir(parents=True, exist_ok=True)

    # Process training set
    train_path = data_dir / "train"
    _process_folder(train_path, aug_train_path)

    # Process validation set
    val_path = data_dir / "val"
    _process_folder(val_path, aug_val_path)

    print(
        f"Data augmentation completed! Augmented dataset saved to: {augmented_data_dir}"
    )
    return augmented_data_dir


def _process_folder(src_folder, dst_folder):
    """Process a single folder for data augmentation"""
    images = [i for i in os.listdir(src_folder) if i.endswith(".jpg")]

    for img_name in images:
        txt_name = img_name.replace(".jpg", ".txt")
        img_path = src_folder / img_name
        txt_path = src_folder / txt_name

        # Read the original image and flip horizontally
        img = cv2.imread(str(img_path))
        flipped_img = cv2.flip(img, 1)  # 1 means horizontal flip

        # Read the original label file
        with open(txt_path, "r") as f:
            content = f.read().strip().split()
            # Invert the steering value (first float)
            turning_value = float(content[0])
            acceleration_value = float(content[1])
            speed_value = content[2] if len(content) > 2 else None

            # Invert steering value
            flipped_turning = -turning_value

        # Save the flipped image
        flipped_img_path = dst_folder / img_name
        cv2.imwrite(str(flipped_img_path), flipped_img)

        # Save the modified label
        flipped_txt_path = dst_folder / txt_name
        with open(flipped_txt_path, "w") as f:
            if speed_value:
                f.write(f"{flipped_turning:.2f} {acceleration_value:.2f} {speed_value}")
            else:
                f.write(f"{flipped_turning:.2f} {acceleration_value:.2f}")

    print(f"Processed {len(images)} images in {src_folder}")


def main(cgl, monitor, keyboardmonitor, RAW_DATA_DIR):
    if not RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    counter = 1
    while True:
        new_data_dir = RAW_DATA_DIR / f"data_{counter}"
        if not new_data_dir.exists():
            new_data_dir.mkdir(parents=True, exist_ok=True)
            break
        counter += 1
    RAW_DATA_DIR = new_data_dir
    digit_data_dir = RAW_DATA_DIR / "digit_data"
    digit_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving data to {RAW_DATA_DIR}")
    print(f"Saving digit data to {digit_data_dir}")

    monitor.start_controller_monitoring()
    # monitor.start_keyboard_monitoring()
    keyboardmonitor.start_monitoring()

    # wait for player to start the game
    time.sleep(5)

    controller_state = {
        "turning": 0.0,
        "acceleration": 0.0,
        }
    frame_label = 0
    while True:
        # get digit region and blue bird eye view
        begin_of_get_frame = time.time()
        digit_region, blue_bird_eye_view = cgl.get_currunt_key_region()
        # cv2.imshow("digit_region", digit_region)
        # cv2.imshow("blue_bird_eye_view", blue_bird_eye_view)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        # save the frame and control state
        cv2.imwrite(str(RAW_DATA_DIR / f"{frame_label}.jpg"), blue_bird_eye_view)
        cv2.imwrite(str(digit_data_dir / f"{frame_label}.jpg"), digit_region)
        with open(RAW_DATA_DIR / f"{frame_label}.txt", "w") as f:
            f.write(
                f"{controller_state['turning']:.2f} {controller_state['acceleration']:.2f}"
            )
        frame_label += 1

        end_of_get_frame = time.time()

        # wait for the next frame
        if (end_of_get_frame - begin_of_get_frame) < 0.05:
            time.sleep(0.05 - (end_of_get_frame - begin_of_get_frame))

        # get contorl state
        controller_state = monitor.get_combined_state()
        # print(f"Turning: {controller_state['turning']:.2f}, Acceleration: {controller_state['acceleration']:.2f}")

        if keyboardmonitor.is_interrupted():
            print("Keyboard interrupt detected. Exiting...")
            break

    # delete the last frame in 5 seconds
    delete_frame_number = int(5 / 0.05)
    for i in range(frame_label - delete_frame_number, frame_label):
        os.remove(str(RAW_DATA_DIR / f"{i}.jpg"))
        os.remove(str(digit_data_dir / f"{i}.jpg"))
        os.remove(str(RAW_DATA_DIR / f"{i}.txt"))

    # stop monitoring
    monitor.stop_monitoring()
    keyboardmonitor.stop_monitoring()
    lite_digit_detector = LiteDigitDetector(input_height=48, input_width=80)

    # Load pretrained weights
    model_path = ROOT_DIR / "model" / "LDD" / "best_digit_model.pth"
    try:
        lite_digit_detector.load_weights(str(model_path))
        print("Successfully loaded model weights")
    except:
        print("Pretrained model weights not found, using randomly initialized model")

    # Set model to evaluation mode
    lite_digit_detector.eval()

    digits = [str(digit_data_dir) + "\\" + i for i in os.listdir(digit_data_dir)]
    for digit in digits:
        print(f"Processing {digit}...")
        img = cv2.imread(digit)
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply median filter to remove noise
        median = cv2.medianBlur(img, 3)

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
        pil_img = Image.fromarray(morph)

        digit_value = lite_digit_detector.predict(pil_img)
        digit_value = digit_value[0] * 100 + digit_value[1] * 10 + digit_value[2]
        with open(digit.replace("digit_data", "").replace(".jpg", ".txt"), "a") as f:
            f.write(f" {digit_value}")
    shutil.rmtree(digit_data_dir)

    # split the data into train and test set
    processed_images = [i for i in os.listdir(RAW_DATA_DIR) if i.endswith((".jpg"))]

    train_path = RAW_DATA_DIR / "train"
    val_path = RAW_DATA_DIR / "val"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    for image in processed_images:
        image_sequence = int(image.split(".")[0])
        if image_sequence < int(len(processed_images) * 0.8):
            shutil.move(str(RAW_DATA_DIR / image), str(train_path / image))
            shutil.move(
                str(RAW_DATA_DIR / image.replace(".jpg", ".txt")),
                str(train_path / image.replace(".jpg", ".txt")),
            )
        else:
            shutil.move(str(RAW_DATA_DIR / image), str(val_path / image))
            shutil.move(
                str(RAW_DATA_DIR / image.replace(".jpg", ".txt")),
                str(val_path / image.replace(".jpg", ".txt")),
            )

    # Data augmentation step
    augmented_data_dir = augment_dataset(RAW_DATA_DIR)
    print(f"Original dataset: {RAW_DATA_DIR}")
    print(f"Augmented dataset: {augmented_data_dir}")

    print(f"Stopping... You have been playing for {frame_label * 0.05:.2f} seconds")


if __name__ == "__main__":
    cgl = CaptureGuideline()
    monitor = InputMonitor()
    km = KeyboardInterrupt()
    main(cgl, monitor, km, RAW_DATA_DIR)
