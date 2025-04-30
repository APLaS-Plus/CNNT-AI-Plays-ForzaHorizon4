import sys
import pathlib
import time
import numpy as np

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
UTILS_DIR = ROOT_DIR / ".."
UTILS_DIR = str(UTILS_DIR.resolve())

sys.path.append(UTILS_DIR)
from utils.controller_output import ControllerOutput

if __name__ == "__main__":
    controller = ControllerOutput()
    time.sleep(2.0)
    controller.start()

    left_trigger_array = [i / 200 for i in range(0, 201)]
    right_trigger_array = [i + 50 / 250 for i in range(0, 201)]
    left_stick_x_array = [-i / 200 for i in range(0, 201)]
    left_stick_x_array2 = [i / 200 for i in range(0, 201)]

    # Test left trigger
    for value in left_trigger_array:
        # print(f"Left Trigger Value: {value}")
        controller.set_controls(left_trigger=value)
        time.sleep(0.01)

    # Test right trigger
    for value in right_trigger_array:
        # print(f"Right Trigger Value: {value}")
        controller.set_controls(right_trigger=value)
        time.sleep(0.01)

    # Test left stick x-axis
    for value in left_stick_x_array:
        # print(f"Left Stick X Value: {value}")
        controller.set_controls(left_stick_x=value)
        time.sleep(0.01)

    # Test left stick x-axis (reverse direction)
    for value in left_stick_x_array2:
        # print(f"Left Stick X Value (Reverse): {value}")
        controller.set_controls(left_stick_x=value)
        time.sleep(0.01)

    controller.stop()
    print("Controller output test completed.")
