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
    lite_digit_detector = LiteDigitDetector(input_height=48, input_width=96)

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
        pil_img = Image.fromarray(morph, mode="L")

        digit_value = lite_digit_detector.predict(pil_img)
        digit_value = digit_value[0] * 100 + digit_value[1] * 10 + digit_value[2]
        with open(digit.replace("digit_data", "").replace(".jpg", ".txt"), "a") as f:
            f.write(f" {digit_value}")

        # 保存处理后的数字图片到原目录中，添加 "_processed" 后缀以区分原图
        processed_img_path = digit.replace(".jpg", "_processed.jpg")
        cv2.imwrite(processed_img_path, morph)

    # 不再删除数字数据目录
    print(f"Digit data preserved at: {digit_data_dir}")

    # 将数据集拆分为4个子集，并按照3:1的比例划分训练集和验证集
    processed_images = [
        i
        for i in os.listdir(RAW_DATA_DIR)
        if i.endswith((".jpg")) and not i.endswith("_processed.jpg")
    ]

    # 按照序列号排序图片
    processed_images.sort(key=lambda x: int(x.split(".")[0]))
    total_images = len(processed_images)

    # 创建4个数据集目录，命名为CNNT_n_m格式
    dataset_dirs = []
    data_dir_name = RAW_DATA_DIR.name  # e.g. "data_1"

    for i in range(4):
        dataset_name = f"{data_dir_name}_{i+1}"
        dataset_dir = RAW_DATA_DIR.parent / dataset_name
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)

        # 为每个数据集创建训练集和验证集目录
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

    # 将图片平均分配到4个数据集
    images_per_dataset = total_images // 4

    for i in range(4):
        start_idx = i * images_per_dataset
        end_idx = (i + 1) * images_per_dataset if i < 3 else total_images
        dataset_images = processed_images[start_idx:end_idx]

        # 按照3:1的比例划分训练集和验证集
        train_split = int(len(dataset_images) * 0.75)

        # 获取当前数据集的目录信息
        current_dataset = dataset_dirs[i]

        # 处理训练集
        for j in range(train_split):
            image = dataset_images[j]
            image_sequence = int(image.split(".")[0])
            digit_image = os.path.join(str(digit_data_dir), image)
            digit_image_processed = os.path.join(
                str(digit_data_dir), image.replace(".jpg", "_processed.jpg")
            )

            # 检查对应的数字图像是否存在
            has_digit_image = os.path.exists(digit_image)
            has_digit_processed = os.path.exists(digit_image_processed)

            # 复制原始图像和对应的文本到训练集
            shutil.copy(
                str(RAW_DATA_DIR / image), str(current_dataset["train"] / image)
            )
            shutil.copy(
                str(RAW_DATA_DIR / image.replace(".jpg", ".txt")),
                str(current_dataset["train"] / image.replace(".jpg", ".txt")),
            )

            # 复制对应的数字图像(如果存在)
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

        # 处理验证集
        for j in range(train_split, len(dataset_images)):
            image = dataset_images[j]
            image_sequence = int(image.split(".")[0])
            digit_image = os.path.join(str(digit_data_dir), image)
            digit_image_processed = os.path.join(
                str(digit_data_dir), image.replace(".jpg", "_processed.jpg")
            )

            # 检查对应的数字图像是否存在
            has_digit_image = os.path.exists(digit_image)
            has_digit_processed = os.path.exists(digit_image_processed)

            # 复制原始图像和对应的文本到验证集
            shutil.copy(str(RAW_DATA_DIR / image), str(current_dataset["val"] / image))
            shutil.copy(
                str(RAW_DATA_DIR / image.replace(".jpg", ".txt")),
                str(current_dataset["val"] / image.replace(".jpg", ".txt")),
            )

            # 复制对应的数字图像(如果存在)
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
        # 为每个数据集执行数据增强
        augmented_data_dir = augment_dataset(dataset["dir"])
        print(f"Dataset: {dataset['name']}")
        print(f"  - Original: {dataset['dir']}")
        print(f"  - Augmented: {augmented_data_dir}")

    print(f"Stopping... You have been playing for {frame_label * 0.05:.2f} seconds")


if __name__ == "__main__":
    cgl = CaptureGuideline()
    monitor = InputMonitor()
    km = KeyboardInterrupt()
    main(cgl, monitor, km, RAW_DATA_DIR)
