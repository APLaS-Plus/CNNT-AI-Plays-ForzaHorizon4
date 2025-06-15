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
    Augment the dataset by horizontal flipping for improved model generalization.

    This function creates mirror images of the original dataset by horizontally
    flipping images and inverting corresponding steering values. This data
    augmentation technique helps the model learn to handle both left and right
    turns equally well, improving generalization.

    Data augmentation process:
    1. Horizontally flip each image (mirror effect)
    2. Invert steering values (left turn becomes right turn)
    3. Keep acceleration and speed values unchanged
    4. Maintain the same file structure

    Args:
        data_dir (Path): Original dataset directory containing train/val splits

    Returns:
        Path: Directory path of the augmented dataset
    """
    print(f"Starting dataset augmentation with horizontal flipping...")

    # Create directory for augmented dataset with descriptive suffix
    augmented_data_dir = data_dir.parent / f"{data_dir.name}_augmented"
    if not augmented_data_dir.exists():
        augmented_data_dir.mkdir(parents=True, exist_ok=True)

    # Create train and validation subdirectories for augmented data
    aug_train_path = augmented_data_dir / "train"
    aug_val_path = augmented_data_dir / "val"
    aug_train_path.mkdir(parents=True, exist_ok=True)
    aug_val_path.mkdir(parents=True, exist_ok=True)

    # Process training set with augmentation
    train_path = data_dir / "train"
    _process_folder(train_path, aug_train_path)

    # Process validation set with augmentation
    val_path = data_dir / "val"
    _process_folder(val_path, aug_val_path)

    print(
        f"Data augmentation completed! Augmented dataset saved to: {augmented_data_dir}"
    )
    return augmented_data_dir


def _process_folder(src_folder, dst_folder):
    """
    Process a single folder for data augmentation with horizontal flipping.

    This helper function handles the actual image processing and label modification
    for data augmentation. It reads each image, applies horizontal flipping,
    adjusts the steering values, and saves the augmented data.

    Args:
        src_folder (Path): Source folder containing original images and labels
        dst_folder (Path): Destination folder for augmented data
    """
    # Get all image files from the source directory
    images = [i for i in os.listdir(src_folder) if i.endswith(".jpg")]

    for img_name in images:
        txt_name = img_name.replace(".jpg", ".txt")
        img_path = src_folder / img_name
        txt_path = src_folder / txt_name

        # Read the original image and apply horizontal flip transformation
        img = cv2.imread(str(img_path))
        flipped_img = cv2.flip(img, 1)  # 1 means horizontal flip along y-axis

        # Read the original label file containing control data
        with open(txt_path, "r") as f:
            content = f.read().strip().split()
            # Extract control values: steering, acceleration, speed
            turning_value = float(content[0])
            acceleration_value = float(content[1])
            speed_value = content[2] if len(content) > 2 else None

            # Invert steering value for horizontal flip consistency
            # Left turn (-) becomes right turn (+) and vice versa
            flipped_turning = -turning_value

        # Save the horizontally flipped image
        flipped_img_path = dst_folder / img_name
        cv2.imwrite(str(flipped_img_path), flipped_img)

        # Save the modified label with inverted steering
        flipped_txt_path = dst_folder / txt_name
        with open(flipped_txt_path, "w") as f:
            if speed_value:
                f.write(f"{flipped_turning:.2f} {acceleration_value:.2f} {speed_value}")
            else:
                f.write(f"{flipped_turning:.2f} {acceleration_value:.2f}")

    print(f"Processed {len(images)} images in {src_folder}")


def main(cgl, monitor, keyboardmonitor, RAW_DATA_DIR):
    """
    Main data collection and processing pipeline for CNNT autonomous driving dataset.

    This function orchestrates the complete data collection process:
    1. Real-time game data capture (images + controls)
    2. Speed digit recognition using pre-trained model
    3. Dataset splitting and organization
    4. Data augmentation for improved training

    The pipeline captures synchronized data at 20fps and processes it for training
    temporal CNN-Transformer models for autonomous driving control prediction.

    Args:
        cgl: CaptureGuideline instance for screen capture and image processing
        monitor: InputMonitor instance for controller input monitoring
        keyboardmonitor: KeyboardInterrupt instance for manual stop control
        RAW_DATA_DIR: Base directory for saving raw collected data
    """
    # Create unique data directory for this collection session
    if not RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Find next available data directory number
    counter = 1
    while True:
        new_data_dir = RAW_DATA_DIR / f"data_{counter}"
        if not new_data_dir.exists():
            new_data_dir.mkdir(parents=True, exist_ok=True)
            break
        counter += 1
    RAW_DATA_DIR = new_data_dir

    # Create dedicated directory for speed digit images
    digit_data_dir = RAW_DATA_DIR / "digit_data"
    digit_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving main data to {RAW_DATA_DIR}")
    print(f"Saving digit data to {digit_data_dir}")

    # Initialize monitoring systems
    monitor.start_controller_monitoring()
    keyboardmonitor.start_monitoring()

    # Wait for player to start the game and stabilize
    time.sleep(5)

    # Initialize control state tracking
    controller_state = {
        "turning": 0.0,
        "acceleration": 0.0,
    }
    frame_label = 0

    # Main data collection loop - captures at 20fps (0.05s intervals)
    while True:
        # Capture current frame and extract key regions
        begin_of_get_frame = time.time()
        digit_region, adjusted_view = cgl.get_currunt_key_region()

        # Optional: Display captured regions for debugging
        # cv2.imshow("digit_region", digit_region)
        # cv2.imshow("blue_bird_eye_view", blue_bird_eye_view)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        # Save synchronized frame data and control state
        cv2.imwrite(str(RAW_DATA_DIR / f"{frame_label}.jpg"), adjusted_view)
        cv2.imwrite(str(digit_data_dir / f"{frame_label}.jpg"), digit_region)

        # Save control state (steering and acceleration) to label file
        with open(RAW_DATA_DIR / f"{frame_label}.txt", "w") as f:
            f.write(
                f"{controller_state['turning']:.2f} {controller_state['acceleration']:.2f}"
            )
        frame_label += 1

        end_of_get_frame = time.time()

        # Maintain precise 20fps timing (0.05s per frame)
        if (end_of_get_frame - begin_of_get_frame) < 0.05:
            time.sleep(0.05 - (end_of_get_frame - begin_of_get_frame))

        # Update control state from input monitoring
        controller_state = monitor.get_combined_state()

        # Check for manual stop signal
        if keyboardmonitor.is_interrupted():
            print("Keyboard interrupt detected. Exiting...")
            break

    # Clean up the last 5 seconds of data (removes potential inconsistencies)
    delete_frame_number = int(5 / 0.05)
    for i in range(frame_label - delete_frame_number, frame_label):
        os.remove(str(RAW_DATA_DIR / f"{i}.jpg"))
        os.remove(str(digit_data_dir / f"{i}.jpg"))
        os.remove(str(RAW_DATA_DIR / f"{i}.txt"))

    # Stop all monitoring systems
    monitor.stop_monitoring()
    keyboardmonitor.stop_monitoring()

    # Initialize speed digit detection model
    lite_digit_detector = LiteDigitDetector(input_height=48, input_width=96)

    # Load pre-trained digit recognition weights
    model_path = ROOT_DIR / "model" / "LDD" / "best_digit_model.pth"
    try:
        lite_digit_detector.load_weights(str(model_path))
        print("Successfully loaded pre-trained digit model weights")
    except:
        print("Pre-trained model weights not found, using randomly initialized model")

    # Set model to evaluation mode for inference
    lite_digit_detector.eval()

    # Process all collected digit images for speed recognition
    digits = [str(digit_data_dir) + "\\" + i for i in os.listdir(digit_data_dir)]
    for digit in digits:
        print(f"Processing digit image: {digit}...")
        img = cv2.imread(digit)

        # Convert to grayscale for better digit recognition
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply image enhancement pipeline for better digit recognition

        # Apply median filter to remove noise
        median = cv2.medianBlur(img, 3)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(median)

        # Apply adaptive thresholding for binary image
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
        )

        # Morphological operations to enhance character contours
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Convert processed image to PIL format for model input
        pil_img = Image.fromarray(morph, mode="L")

        # Predict speed digits and convert to numerical value
        digit_value = lite_digit_detector.predict(pil_img)
        digit_value = digit_value[0] * 100 + digit_value[1] * 10 + digit_value[2]

        # Append speed value to the corresponding label file
        with open(digit.replace("digit_data", "").replace(".jpg", ".txt"), "a") as f:
            f.write(f" {digit_value}")

        # Save processed digit image for analysis (with "_processed" suffix)
        processed_img_path = digit.replace(".jpg", "_processed.jpg")
        cv2.imwrite(processed_img_path, morph)

    # Preserve digit data directory for future analysis
    print(f"Digit data preserved at: {digit_data_dir}")

    # Split dataset into 4 subsets with 3:1 train-validation ratio
    processed_images = [
        i
        for i in os.listdir(RAW_DATA_DIR)
        if i.endswith((".jpg")) and not i.endswith("_processed.jpg")
    ]

    # Sort images by sequence number to maintain temporal order
    processed_images.sort(key=lambda x: int(x.split(".")[0]))
    total_images = len(processed_images)

    # Create 4 dataset directories with structured naming convention
    dataset_dirs = []
    data_dir_name = RAW_DATA_DIR.name  # e.g. "data_1"

    for i in range(4):
        dataset_name = f"{data_dir_name}_{i+1}"
        dataset_dir = RAW_DATA_DIR.parent / dataset_name
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create train and validation subdirectories for each dataset
        train_path = dataset_dir / "train"
        val_path = dataset_dir / "val"
        train_digit_path = train_path / "digit_data"
        val_digit_path = val_path / "digit_data"

        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)
        train_digit_path.mkdir(parents=True, exist_ok=True)
        val_digit_path.mkdir(parents=True, exist_ok=True)

        dataset_dirs.append(
            {
                "name": dataset_name,
                "dir": dataset_dir,
                "train": train_path,
                "val": val_path,
                "train_digit": train_digit_path,
                "val_digit": val_digit_path,
            }
        )

    # Distribute images evenly across 4 datasets
    images_per_dataset = total_images // 4

    for i in range(4):
        start_idx = i * images_per_dataset
        end_idx = (i + 1) * images_per_dataset if i < 3 else total_images
        dataset_images = processed_images[start_idx:end_idx]

        # Split current dataset into 75% training, 25% validation
        train_split = int(len(dataset_images) * 0.75)

        # Get current dataset directory structure
        current_dataset = dataset_dirs[i]

        # Process training split
        for j in range(train_split):
            image = dataset_images[j]
            image_sequence = int(image.split(".")[0])
            digit_image = os.path.join(str(digit_data_dir), image)
            digit_image_processed = os.path.join(
                str(digit_data_dir), image.replace(".jpg", "_processed.jpg")
            )

            # Check if corresponding digit images exist
            has_digit_image = os.path.exists(digit_image)
            has_digit_processed = os.path.exists(digit_image_processed)

            # Copy main image and label to training set
            shutil.copy(
                str(RAW_DATA_DIR / image), str(current_dataset["train"] / image)
            )
            shutil.copy(
                str(RAW_DATA_DIR / image.replace(".jpg", ".txt")),
                str(current_dataset["train"] / image.replace(".jpg", ".txt")),
            )

            # Copy corresponding digit images if they exist
            if has_digit_image:
                shutil.copy(digit_image, str(current_dataset["train_digit"] / image))
            if has_digit_processed:
                shutil.copy(
                    digit_image_processed,
                    str(
                        current_dataset["train_digit"]
                        / image.replace(".jpg", "_processed.jpg")
                    ),
                )

        # Process validation split
        for j in range(train_split, len(dataset_images)):
            image = dataset_images[j]
            image_sequence = int(image.split(".")[0])
            digit_image = os.path.join(str(digit_data_dir), image)
            digit_image_processed = os.path.join(
                str(digit_data_dir), image.replace(".jpg", "_processed.jpg")
            )

            # Check if corresponding digit images exist
            has_digit_image = os.path.exists(digit_image)
            has_digit_processed = os.path.exists(digit_image_processed)

            # Copy main image and label to validation set
            shutil.copy(str(RAW_DATA_DIR / image), str(current_dataset["val"] / image))
            shutil.copy(
                str(RAW_DATA_DIR / image.replace(".jpg", ".txt")),
                str(current_dataset["val"] / image.replace(".jpg", ".txt")),
            )

            # Copy corresponding digit images if they exist
            if has_digit_image:
                shutil.copy(digit_image, str(current_dataset["val_digit"] / image))
            if has_digit_processed:
                shutil.copy(
                    digit_image_processed,
                    str(
                        current_dataset["val_digit"]
                        / image.replace(".jpg", "_processed.jpg")
                    ),
                )

    print("Data has been split into 4 datasets with 3:1 train-validation ratio:")
    for dataset in dataset_dirs:
        # Perform data augmentation for each dataset to improve generalization
        augmented_data_dir = augment_dataset(dataset["dir"])
        print(f"Dataset: {dataset['name']}")
        print(f"  - Original: {dataset['dir']}")
        print(f"  - Augmented: {augmented_data_dir}")

    print(
        f"Data collection completed! Total recording time: {frame_label * 0.05:.2f} seconds"
    )


if __name__ == "__main__":
    # Initialize all required components for data collection
    cgl = CaptureGuideline()  # Screen capture and image processing
    monitor = InputMonitor()  # Controller input monitoring
    km = KeyboardInterrupt()  # Keyboard interrupt handling

    # Start the main data collection and processing pipeline
    main(cgl, monitor, km, RAW_DATA_DIR)
