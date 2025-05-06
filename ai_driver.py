import sys
import pathlib
import time
import cv2
import torch
from torchvision import transforms
import numpy as np
import math
from PIL import Image
import threading

ROOT_DIR = pathlib.Path(__file__).resolve().parent

UTILS_DIR = ROOT_DIR
UTILS_DIR = str(UTILS_DIR.resolve())

sys.path.append(UTILS_DIR)
from utils.capture_guideline import CaptureGuideline
from utils.controller_output import ControllerOutput
from utils.model.CNN_GLtransformer import CNNT
from utils.model.lite_digit_detector import LiteDigitDetector


def sanitize_output(value):
    if math.isnan(value) or math.isinf(value):
        return 0.0

    if abs(value) > 0.999999:
        return math.copysign(0.999999, value)

    return value


def debug_output(value):
    """Detailed examination of potentially problematic output values"""
    if torch.is_tensor(value):
        torch_value = value.item()
    else:
        torch_value = value

    print(f"Value: {value}")
    print(f"Type: {type(value)}")
    print(
        f"torch.isnan: {torch.isnan(torch.tensor(value)) if torch.is_tensor(value) else torch.isnan(torch.tensor([value]))[0]}"
    )
    print(
        f"numpy.isnan: {np.isnan(value) if isinstance(value, (float, int, np.number)) else np.isnan(np.array(value))}"
    )
    print(
        f"math.isnan: {math.isnan(float(value)) if not isinstance(value, str) else 'N/A'}"
    )
    print(f"Value after float conversion: {float(value)}")
    print(
        f"Value hex representation: {value.hex() if isinstance(value, float) else 'N/A'}"
    )


class AIDriver:
    """
    AI driver for Forza Horizon 4 using CNNT model for control prediction
    and virtual controller for game input.
    """

    def __init__(self):
        # Initialize screen capture
        self.cgl = CaptureGuideline()

        # Initialize controller output
        self.controller = ControllerOutput()

        # Initialize digit detector
        self.digit_detector = LiteDigitDetector(input_height=48, input_width=80)
        self.load_digit_detector()

        # Initialize CNNT model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = CNNT(
            input_height=240, input_width=136, maxtime_step=40, memory_size=160
        )
        self.load_cnnt_model()

        # Initialize memory queues for sequential prediction
        self.frame_queue = None
        self.speed_queue = None
        self.memory_queue = None

        # Control flags
        self.running = False
        self.thread = None

        # Performance tracking
        self.frame_times = []
        self.last_time = time.time()

    def load_digit_detector(self):
        """Load the digit detector model"""
        model_path = ROOT_DIR / "model" / "LDD" / "best_digit_model.pth"
        try:
            self.digit_detector.load_weights(str(model_path))
            print("Successfully loaded digit detector model weights")
        except Exception as e:
            print(f"Error loading digit detector: {e}")
            print("Using randomly initialized digit detector model")
        self.digit_detector.eval()

    def load_cnnt_model(self):
        """Load the CNNT model"""
        model_path = ROOT_DIR / "model" / "CNNT" / "best_cnnt_model.pth"
        try:
            self.model.load_state_dict(
                torch.load(str(model_path), map_location=self.device)
            )
            print("Successfully loaded CNNT model weights")
        except Exception as e:
            print(f"Error loading CNNT model: {e}")
            print("Using randomly initialized CNNT model")
        self.model.to(self.device)
        self.model.eval()

    def detect_speed(self, digit_region):
        """Detect speed from digit region using the digit detector"""
        if digit_region is None:
            return torch.tensor([[0.0]], device=self.device)

        # Convert to grayscale
        img = cv2.cvtColor(digit_region, cv2.COLOR_BGR2GRAY)

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

        # Predict digits
        with torch.no_grad():
            digits = self.digit_detector.predict(pil_img)

        # Calculate speed value
        speed_value = digits[0] * 100 + digits[1] * 10 + digits[2]
        return torch.tensor([[float(speed_value)]], device=self.device)

    def process_frame(self):
        """Process a single frame: capture, detect speed, predict controls"""
        # Get current state from screen
        digit_region, blue_bird_eye_view = self.cgl.get_currunt_key_region()

        # cv2.imshow("digit_region", digit_region)
        # cv2.imshow("blue_bird_eye_view", blue_bird_eye_view)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     self.running = False
        #     return

        # Detect speed from digit region
        speed = self.detect_speed(digit_region)

        # Convert guideline image for model input
        # print(blue_bird_eye_view.shape)
        guideline_tensor = Image.fromarray(blue_bird_eye_view).convert("L")
        transform = transforms.Compose(
            [
                transforms.Resize((240, 136)),
                transforms.ToTensor(),
            ]
        )
        guideline_tensor = transform(guideline_tensor).unsqueeze(
            0
        )  # Add batch dimension
        # print(guideline_tensor.shape)
        guideline_tensor = guideline_tensor.to(self.device)

        # Model inference
        with torch.no_grad():
            (
                steering,
                acceleration,
                self.memory_queue,
                self.frame_queue,
                self.speed_queue,
            ) = self.model(
                self.frame_queue,
                self.speed_queue,
                guideline_tensor,
                speed,
                self.memory_queue,
                device=self.device,
            )
            print("predict scuessfully")

        # print(
        #     f"steering: {torch.isnan(steering)}, acceleration: {torch.isnan(acceleration)}"
        # )
        steering, acceleration = (
            steering.item(),
            acceleration.item(),
        )
        debug_output(steering)
        debug_output(acceleration)
        # print(f"steering: {np.isnan(steering)}, acceleration: {np.isnan(acceleration)}")
        return steering, acceleration

    def update_controller(self, steering, acceleration):
        """Update controller with steering and acceleration values"""
        # Map model outputs to controller inputs
        left_stick_x = steering  # Steering is already in range [-1, 1]

        # For acceleration, map from [-1, 1] to gas/brake triggers
        if acceleration >= 0:
            right_trigger = acceleration  # Gas (acceleration > 0)
            left_trigger = 0.0  # No brake
        else:
            right_trigger = 0.0  # No gas
            left_trigger = -acceleration  # Brake (acceleration < 0)

        # Set controller values
        self.controller.set_controls(
            left_stick_x=left_stick_x,
            left_trigger=left_trigger,
            right_trigger=right_trigger,
        )

    def drive_loop(self):
        """Main driving loop running at target 20fps"""
        print("AI driver started. Press Ctrl+C to stop.")

        frame_count = 0
        warmup_frames = 20  # Allow some frames for model to warm up

        try:
            while self.running:
                start_time = time.time()

                # Process frame and get control values
                steering, acceleration = self.process_frame()
                steering = sanitize_output(steering)
                acceleration = sanitize_output(acceleration)
                # print(f"steering: {steering}, acceleration: {acceleration}")
                # Only apply controls after warmup period

                if frame_count >= warmup_frames:
                    self.update_controller(steering, acceleration)
                    print(
                        f"Frame {frame_count}: Speed: {self.speed_queue[:, -1].item():.1f}, Steering: {steering:.3f}, Acceleration: {acceleration:.3f}"
                    )

                # Calculate processing time
                process_time = time.time() - start_time
                self.frame_times.append(process_time)

                # Sleep to maintain target frame rate (20fps = 0.05s per frame)
                if process_time < 0.1:
                    time.sleep(0.1 - process_time)

                frame_count += 1

        except KeyboardInterrupt:
            print("Keyboard interrupt received, stopping driver")
        except Exception as e:
            print(f"Error in drive loop: {e}")
        finally:
            # Calculate average FPS
            if self.frame_times:
                avg_process_time = sum(self.frame_times) / len(self.frame_times)
                avg_fps = 1.0 / avg_process_time
                print(f"Average processing time: {avg_process_time:.4f}s")
                print(f"Average FPS: {avg_fps:.2f}")

            # Reset controller state
            self.controller.set_controls(0.0, 0.0, 0.0)
            self.controller.stop()
            self.running = False

    def start(self):
        """Start the AI driver in a separate thread"""
        if self.thread is not None and self.thread.is_alive():
            print("AI driver is already running")
            return

        print("Starting AI driver...")
        self.running = True
        self.controller.start()

        # Start in a separate thread
        self.thread = threading.Thread(target=self.drive_loop)
        self.thread.daemon = True
        self.thread.start()

        # Wait for user to stop with Ctrl+C
        try:
            while self.thread.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping AI driver...")
            self.running = False
            self.thread.join(timeout=2.0)

    def stop(self):
        """Stop the AI driver"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.controller.stop()
        print("AI driver stopped")


def main():
    """Main function to start the AI driver"""
    print("Initializing AI driver for Forza Horizon 4...")
    driver = AIDriver()

    print("Wait 5 seconds to focus on the game window...")
    time.sleep(5)

    driver.start()


if __name__ == "__main__":
    main()
