import sys
import pathlib
import time
import math

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
UTILS_DIR = ROOT_DIR / ".."
UTILS_DIR = str(UTILS_DIR.resolve())

sys.path.append(UTILS_DIR)
from utils.input_monitor import InputMonitor


# Usage example
if __name__ == "__main__":
    monitor = InputMonitor()
    # Choose to monitor controller or keyboard
    monitor.start_controller_monitoring()

    try:
        while True:
            controller_state = monitor.get_controller_state()
            print(
                f"Left Thumbstick X: {controller_state['left_thumb_x']:.2f}, "
                f"Left Trigger: {controller_state['left_trigger']:.2f}, Right Trigger: {controller_state['right_trigger']:.2f}"
            )
            time.sleep(0.1)
            combined_controller_state = monitor.get_combined_state()
            print(
                f"Combined - Turning: {combined_controller_state['turning']:.2f}, "
                f"Acceleration: {combined_controller_state['acceleration']:.2f}"
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
