import sys
import pathlib
import time
import shutil
import cv2
import os
from PIL import Image
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

UTILS_DIR = ROOT_DIR
UTILS_DIR = str(UTILS_DIR.resolve())

sys.path.append(UTILS_DIR)
from utils.capture_guideline import CaptureGuideline
from utils.input_monitor import InputMonitor, KeyboardInterrupt
from utils.model.lite_digit_detector import LiteDigitDetector

RAW_DATA_DIR = ROOT_DIR / "raw_data" / "CNNT"

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

    frame_label = 0
    while True:
        # get digit region and blue bird eye view
        begin_of_get_frame = time.time()
        digit_region, blue_bird_eye_view = cgl.get_currunt_key_region()
        cv2.imshow("digit_region", digit_region)
        cv2.imshow("blue_bird_eye_view", blue_bird_eye_view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # get contorl state
        controller_state = monitor.get_combined_state()
        # print(f"Turning: {controller_state['turning']:.2f}, Acceleration: {controller_state['acceleration']:.2f}")

        # save the frame and control state
        cv2.imwrite(str(RAW_DATA_DIR / f"{frame_label}.jpg"), blue_bird_eye_view)
        cv2.imwrite(str(digit_data_dir / f"{frame_label}.jpg"), digit_region)
        with open(RAW_DATA_DIR / f"{frame_label}.txt", "w") as f:
            f.write(f"{controller_state['turning']:.2f} {controller_state['acceleration']:.2f}")
        frame_label += 1

        end_of_get_frame = time.time()

        # wait for the next frame
        if(end_of_get_frame - begin_of_get_frame) < 0.05:
            time.sleep(0.05 - (end_of_get_frame - begin_of_get_frame))

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

    digits = [str(digit_data_dir)+ "\\" + i for i in os.listdir(digit_data_dir)]
    for digit in digits:
        print(f"Processing {digit}...")
        img = cv2.imread(digit)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(img)
        digit_value = lite_digit_detector.predict(pil_img)
        digit_value = digit_value[0] * 100 + digit_value[1] * 10 + digit_value[2]
        with open(digit.replace("digit_data", "").replace(".jpg", ".txt"), "a") as f:
            f.write(f" {digit_value}")
    shutil.rmtree(digit_data_dir)

    # split the data into train and test set
    processed_images = [i for i in os.listdir(RAW_DATA_DIR) if i.endswith(('.jpg'))]

    train_path = RAW_DATA_DIR / "train"
    val_path = RAW_DATA_DIR / "val"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    for image in processed_images:
        image_sequence = int(image.split(".")[0])
        if image_sequence < int(len(processed_images) * 0.8):
            shutil.move(str(RAW_DATA_DIR / image), str(train_path / image))
            shutil.move(str(RAW_DATA_DIR / image.replace('.jpg', '.txt')), str(train_path / image.replace('.jpg', '.txt')))
        else:
            shutil.move(str(RAW_DATA_DIR / image), str(val_path / image))
            shutil.move(str(RAW_DATA_DIR / image.replace('.jpg', '.txt')), str(val_path / image.replace('.jpg', '.txt')))
    


if __name__ == "__main__":
    cgl = CaptureGuideline()
    monitor = InputMonitor()
    km = KeyboardInterrupt()
    main(cgl, monitor, km, RAW_DATA_DIR)
