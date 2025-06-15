import cv2
import numpy as np
import os
import pathlib
import easyocr
import shutil
import multiprocessing
import time

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
RAW_DATASET_DIR = ROOT_DIR / ".." / "raw_data" / "digit"
DATASET_DIR = ROOT_DIR / ".." / "dataset" / "digit"
MODEL_PATH = MODEL_PATH = ROOT_DIR / ".." / "model" / "easyocr"
# print("Root directory:", ROOT_DIR)


def capture_digit_region(img):
    """
    Capture the digit region from the image.
    The region is defined as a rectangle with fixed coordinates.
    """
    # Define the coordinates for the digit region (x, y, width, height)
    x, y, w, h = 1710, 912, 160, 96
    # Crop the image to get the digit region
    digit_region = img[y : y + h, x : x + w]
    return digit_region


if __name__ == "__main__":
    bg = time.time()
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR, exist_ok=True)

    easyocr_reader = easyocr.Reader(
        gpu=True, lang_list=["en"], model_storage_directory=str(MODEL_PATH)
    )

    load_ocr_model = time.time()

    try:
        # Get the list of image files in the raw dataset directory
        image_files = [f for f in os.listdir(RAW_DATASET_DIR) if f.endswith((".jpg"))]

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the raw dataset directory.")
        exit(1)

    print(f"Found {len(image_files)} images in the raw dataset directory.")
    bg_build = time.time()
    for i in range(len(image_files)):
        img_path = os.path.join(RAW_DATASET_DIR, image_files[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Capture the digit region from the image
        digit_region = capture_digit_region(img)

        # Save the cropped digit region to the dataset directory
        output_path = os.path.join(DATASET_DIR, str(i) + ".jpg")
        cv2.imwrite(output_path, digit_region)

        digit = easyocr_reader.recognize(
            digit_region, batch_size=10, allowlist="0123456789"
        )[0][1]
        
        if len(digit) == 2:
            digit = "0" + digit
        elif len(digit) == 1:
            digit = "00" + digit

        for j in range(len(digit)):
            if digit[j] not in "0123456789":
                print(digit)
                print(
                    f"Warning: {digit} is not a valid digit. Please check the image: {img_path}"
                )

                import time

                time.sleep(10000)

        digit = " ".join(digit)

        # Save the digit label to a text file
        label_path = os.path.splitext(output_path)[0] + ".txt"
        with open(label_path, "w") as f:
            f.write(str(digit))
        print(f"Processed {i+1}/{len(image_files)}: {output_path} -> {digit}")

    processed_images = [i for i in os.listdir(DATASET_DIR) if i.endswith((".jpg"))]
    train_images = processed_images[: int(len(processed_images) * 0.8)]
    val_images = processed_images[int(len(processed_images) * 0.8) :]

    if not os.path.exists(os.path.join(DATASET_DIR, "train")):
        os.makedirs(os.path.join(DATASET_DIR, "train"))
    if not os.path.exists(os.path.join(DATASET_DIR, "val")):
        os.makedirs(os.path.join(DATASET_DIR, "val"))
    for image in train_images:
        shutil.move(
            os.path.join(DATASET_DIR, image), os.path.join(DATASET_DIR, "train", image)
        )
        shutil.move(
            os.path.join(DATASET_DIR, image.replace(".jpg", ".txt")),
            os.path.join(DATASET_DIR, "train", image.replace(".jpg", ".txt")),
        )
    for image in val_images:
        shutil.move(
            os.path.join(DATASET_DIR, image), os.path.join(DATASET_DIR, "val", image)
        )
        shutil.move(
            os.path.join(DATASET_DIR, image.replace(".jpg", ".txt")),
            os.path.join(DATASET_DIR, "val", image.replace(".jpg", ".txt")),
        )

    print(f"Load OCR model time: {load_ocr_model - bg:.2f} seconds")
    print(f"Build dataset time: {bg_build - load_ocr_model:.2f} seconds")
    print(f"Total time: {time.time() - bg:.2f} seconds")
