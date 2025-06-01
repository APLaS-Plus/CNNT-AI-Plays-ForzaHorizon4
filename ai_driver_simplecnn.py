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
from collections import defaultdict

ROOT_DIR = pathlib.Path(__file__).resolve().parent

UTILS_DIR = ROOT_DIR
UTILS_DIR = str(UTILS_DIR.resolve())

sys.path.append(UTILS_DIR)
from utils.capture_guideline import CaptureGuideline
from utils.controller_output import ControllerOutput
from utils.model.simpleCNNbaseline import SimpleCNNBaseline
from utils.model.lite_digit_detector import LiteDigitDetector


# 添加时间测量工具
class TimeMeasurement:
    def __init__(self):
        self.times = defaultdict(list)

    def measure(self, name):
        return TimeContext(self, name)

    def get_stats(self):
        """计算并返回每个部分的平均耗时和百分比"""
        stats = {}
        total_time = 0
        for name, times in self.times.items():
            if times:
                avg_time = sum(times) / len(times)
                stats[name] = {
                    "avg": avg_time,
                    "count": len(times),
                    "total": sum(times),
                }
                total_time += sum(times)

        # 计算百分比
        if total_time > 0:
            for name in stats:
                stats[name]["percentage"] = (stats[name]["total"] / total_time) * 100

        return stats, total_time


class TimeContext:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.timer.times[self.name].append(time.time() - self.start)


def sanitize_output(value):
    if math.isnan(value) or math.isinf(value):
        return 0.0

    if abs(value) > 0.999999:
        return math.copysign(0.999999, value)

    return value


class AIDriverSimpleCNN:
    """
    AI driver for Forza Horizon 4 using SimpleCNN model for control prediction
    and virtual controller for game input.
    """

    def __init__(self):
        # Initialize screen capture
        self.cgl = CaptureGuideline()

        # Initialize controller output
        self.controller = ControllerOutput()

        # Initialize digit detector
        self.digit_detector = LiteDigitDetector(input_height=48, input_width=96)
        self.load_digit_detector()

        # Initialize SimpleCNN model with 960*640 input size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNNBaseline(input_w=960, input_h=640)
        self.load_simplecnn_model()

        # Control flags
        self.running = False
        self.thread = None

        # Performance tracking
        self.frame_times = []
        self.last_time = time.time()

        # 添加时间测量工具
        self.timer = TimeMeasurement()

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

    def load_simplecnn_model(self):
        """Load the SimpleCNN model"""
        model_path = ROOT_DIR / "model" / "SimpleCNN" / "best_cnn_model.pth"
        try:
            self.model.load_state_dict(
                torch.load(str(model_path), map_location=self.device)
            )
            print("Successfully loaded SimpleCNN model weights")
        except Exception as e:
            print(f"Error loading SimpleCNN model: {e}")
            print("Using randomly initialized SimpleCNN model")
        self.model.to(self.device)
        self.model.eval()

    def detect_speed(self, digit_region):
        """Detect speed from digit region using the digit detector"""
        if digit_region is None:
            return torch.tensor([[0.0]], device=self.device)

        # Convert to grayscale
        img = cv2.cvtColor(digit_region, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(img, mode="L")
        # # Apply median filter to remove noise
        # median = cv2.medianBlur(img, 3)

        # # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # enhanced = clahe.apply(median)

        # # Apply adaptive thresholding
        # binary = cv2.adaptiveThreshold(
        #     enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
        # )

        # # Morphological operations to enhance character contours
        # kernel = np.ones((2, 2), np.uint8)
        # morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # # Convert processed image to PIL format
        # pil_img = Image.fromarray(morph, mode="L")

        # Predict digits
        with torch.no_grad():
            digits = self.digit_detector.predict(pil_img)

        # Calculate speed value
        speed_value = digits[0] * 100 + digits[1] * 10 + digits[2]
        # Normalize speed to [0, 1] range (assuming max speed around 400)
        normalized_speed = min(speed_value / 400.0, 1.0)
        return torch.tensor([[float(normalized_speed)]], device=self.device)

    def process_frame(self):
        """Process a single frame: capture, detect speed, predict controls"""
        with self.timer.measure("总处理时间"):
            # Get current state from screen
            with self.timer.measure("屏幕捕获"):
                digit_region, adjusted_frame = self.cgl.get_currunt_key_region()

            # Detect speed from digit region
            with self.timer.measure("速度检测"):
                speed = self.detect_speed(digit_region)

            # Convert guideline image for model input - 保持240x144分辨率
            with self.timer.measure("图像预处理"):
                guideline_tensor = Image.fromarray(adjusted_frame).convert("RGB")
                transform = transforms.Compose(
                    [
                        transforms.Resize((640, 960)),  # 保持原始分辨率
                        transforms.ToTensor(),
                    ]
                )
                guideline_tensor = transform(guideline_tensor).unsqueeze(0)
                guideline_tensor = guideline_tensor.to(self.device)

            # Model inference
            with self.timer.measure("模型推理"):
                with torch.no_grad():
                    outputs = self.model(guideline_tensor, speed)
                    steering = outputs[0, 0]  # First output: steering
                    acceleration = outputs[0, 1]  # Second output: acceleration
                    print("predict successfully")

            steering, acceleration = steering.item(), acceleration.item()

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
        warmup_frames = 5  # Reduced warmup frames for SimpleCNN

        try:
            while self.running:
                start_time = time.time()

                # Process frame and get control values
                steering, acceleration = self.process_frame()
                steering = sanitize_output(steering)
                acceleration = sanitize_output(acceleration)

                # Only apply controls after warmup period
                if frame_count >= warmup_frames:
                    self.update_controller(steering, acceleration)
                    # Get current speed for display
                    current_speed = self.detect_speed(
                        self.cgl.get_currunt_key_region()[0]
                    )
                    speed_value = (
                        current_speed.item() * 400.0
                    )  # Denormalize for display
                    print(
                        f"Frame {frame_count}: Speed: {speed_value:.1f}, "
                        f"Steering: {steering:.3f}, Acceleration: {acceleration:.3f}"
                    )

                # Calculate processing time
                process_time = time.time() - start_time
                self.frame_times.append(process_time)

                # Sleep to maintain target frame rate (20fps = 0.05s per frame)
                if process_time < 0.02:
                    time.sleep(0.02 - process_time)

                # 每100帧打印一次耗时统计
                if frame_count > 0 and frame_count % 100 == 0:
                    self.print_time_stats()

                frame_count += 1

        except KeyboardInterrupt:
            print("Keyboard interrupt received, stopping driver")
        except Exception as e:
            print(f"Error in drive loop: {e}")
        finally:
            # 打印最终的耗时统计
            self.print_time_stats()

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

    def print_time_stats(self):
        """打印各模块耗时统计"""
        stats, total_time = self.timer.get_stats()

        print("\n===== SimpleCNN模型性能统计 =====")
        print(f"总耗时: {total_time:.4f}秒")
        print("-" * 40)
        print(f"{'模块名称':<25} {'平均耗时(ms)':<15} {'百分比':<10} {'调用次数'}")
        print("-" * 40)

        # 按百分比降序排序
        sorted_stats = sorted(
            stats.items(), key=lambda x: x[1]["percentage"], reverse=True
        )
        for name, info in sorted_stats:
            print(
                f"{name:<25} {info['avg']*1000:<15.2f} {info['percentage']:<10.2f}% {info['count']}"
            )

        print("=" * 40)

    def start(self):
        """Start the AI driver in a separate thread"""
        if self.thread is not None and self.thread.is_alive():
            print("AI driver is already running")
            return

        print("Starting SimpleCNN AI driver...")
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
    print("Initializing SimpleCNN AI driver for Forza Horizon 4...")
    driver = AIDriverSimpleCNN()

    print("Wait 5 seconds to focus on the game window...")
    time.sleep(5)

    driver.start()


if __name__ == "__main__":
    main()
