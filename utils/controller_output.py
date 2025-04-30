import vgamepad
import threading
import time
from typing import Dict, Any, Optional

class ControllerOutput:
    """
    A class to handle Xbox 360 controller output through vgamepad library.
    Running in a separate thread to avoid blocking the main thread.
    """

    def __init__(self):
        """Initialize the controller and control variables"""
        self.gamepad = vgamepad.VX360Gamepad()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Control values
        self.controls = {
            'left_stick_x': 0.0,  # Range: -1.0 to 1.0
            'left_trigger': 0.0,   # Range: 0.0 to 1.0
            'right_trigger': 0.0,  # Range: 0.0 to 1.0
        }

        # Thread lock for thread-safe control updates
        self.lock = threading.Lock()

    def start(self):
        """Start the controller output thread"""
        if self.thread is not None and self.thread.is_alive():
            return

        self.running = True
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True
        self.thread.start()
        print("Controller output thread started")

    def stop(self):
        """Stop the controller output thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

        # Reset controller state
        self.gamepad.reset()
        self.gamepad.update()
        print("Controller output thread stopped")

    def set_controls(self, left_stick_x: float = 0.0, left_trigger: float = 0.0, right_trigger: float = 0.0):
        """
        Set control values for the controller
        
        Args:
            left_stick_x: X-axis value for left stick (-1.0 to 1.0)
            left_trigger: Value for left trigger (0.0 to 1.0)
            right_trigger: Value for right trigger (0.0 to 1.0)
        """
        with self.lock:
            if left_stick_x is not None:
                self.controls['left_stick_x'] = max(-1.0, min(1.0, left_stick_x))

            if left_trigger is not None:
                self.controls['left_trigger'] = max(0.0, min(1.0, left_trigger))

            if right_trigger is not None:
                self.controls['right_trigger'] = max(0.0, min(1.0, right_trigger))

    def _update_loop(self):
        """The main loop that updates controller state in a separate thread"""
        # print(f"Check running status: {self.running}")
        while self.running:
            # Small sleep to prevent CPU overuse
            time.sleep(0.01)
            # print(f"Check inside loop: {self.running}")
            with self.lock:
                # Apply the current control values to the virtual gamepad
                self.gamepad.left_joystick_float(
                    self.controls['left_stick_x'], 
                    0.0  # We only need x-axis control as per requirements
                )
                self.gamepad.left_trigger_float(self.controls['left_trigger'])
                self.gamepad.right_trigger_float(self.controls['right_trigger'])
                print(f"Updated controls: {self.controls}", flush=True)

            # Update the gamepad state
            self.gamepad.update()

    def __del__(self):
        """Cleanup when object is deleted"""
        self.stop()
        del self.gamepad
        print("Controller resources released")
