import ctypes
import threading
import time
from ctypes import wintypes

# XInput API constants and structures
XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE = 7849
XINPUT_GAMEPAD_TRIGGER_THRESHOLD = 30

# Keyboard virtual key codes
VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
VK_CONTROL = 0x11
VK_C = 0x43


class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", ctypes.c_uint),
        ("Gamepad", XINPUT_GAMEPAD),
    ]


# Load XInput DLL
try:
    xinput = ctypes.windll.xinput1_4
except WindowsError:
    try:
        xinput = ctypes.windll.xinput1_3
    except WindowsError:
        xinput = ctypes.windll.xinput9_1_0


class KeyboardInterrupt:
    def __init__(self):
        self._running = False
        self._interrupt_flag = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
    def start_monitoring(self):
        """Start monitoring for Ctrl+C key combination"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
            
        self._running = True
        self._interrupt_flag = False
        self._monitor_thread = threading.Thread(target=self._monitor_keys)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self._running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
            
    def _monitor_keys(self):
        """Thread function to monitor Ctrl+C key combination"""
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        GetAsyncKeyState = user32.GetAsyncKeyState
        
        while self._running:
            # Check if both Ctrl and C are pressed
            ctrl_pressed = (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0
            c_pressed = (GetAsyncKeyState(VK_C) & 0x8000) != 0
            
            if ctrl_pressed and c_pressed:
                with self._lock:
                    self._interrupt_flag = True
                    
            time.sleep(0.1)  # 100ms polling interval
            
    def is_interrupted(self):
        """Check if Ctrl+C was detected"""
        with self._lock:
            return self._interrupt_flag
            
    def reset(self):
        """Reset the interrupt flag"""
        with self._lock:
            self._interrupt_flag = False


class InputMonitor:
    def __init__(self):
        # Controller state
        self.left_thumb_x = 0.0  # Range [-1.0, 1.0]
        self.left_trigger = 0.0  # Range [0.0, 1.0]
        self.right_trigger = 0.0  # Range [0.0, 1.0]

        # Keyboard state
        self.w_pressed = 0
        self.a_pressed = 0
        self.s_pressed = 0
        self.d_pressed = 0

        # combine state
        self.turning = 0.0  # Range [-1.0, 1.0]
        self.acceleration = 0.0  # Range [-1.0, 1.0]

        # Thread control
        self._running = False
        self._controller_thread = None
        self._keyboard_thread = None
        self._lock = threading.Lock()

    def start_controller_monitoring(self):
        """Start monitoring game controller"""
        if self._controller_thread and self._controller_thread.is_alive():
            return

        self._running = True
        self._controller_thread = threading.Thread(target=self._monitor_controller)
        self._controller_thread.daemon = True
        self._controller_thread.start()

    def start_keyboard_monitoring(self):
        """Start monitoring keyboard"""
        if self._keyboard_thread and self._keyboard_thread.is_alive():
            return

        self._running = True
        self._keyboard_thread = threading.Thread(target=self._monitor_keyboard)
        self._keyboard_thread.daemon = True
        self._keyboard_thread.start()

    def stop_monitoring(self):
        """Stop all monitoring"""
        self._running = False
        if self._controller_thread and self._controller_thread.is_alive():
            self._controller_thread.join(timeout=1.0)
        if self._keyboard_thread and self._keyboard_thread.is_alive():
            self._keyboard_thread.join(timeout=1.0)

    def _monitor_controller(self):
        """Thread function to monitor game controller state"""
        state = XINPUT_STATE()
        controller_id = 0  # Default to first controller

        while self._running:
            result = xinput.XInputGetState(controller_id, ctypes.byref(state))

            if result == 0:  # 0 means success
                # Process left thumbstick X axis
                raw_x = state.Gamepad.sThumbLX

                # Apply deadzone
                thumb_x = self._apply_deadzone(
                    raw_x, XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE
                )

                # Apply trigger threshold
                left_trigger = (
                    max(
                        0, state.Gamepad.bLeftTrigger - XINPUT_GAMEPAD_TRIGGER_THRESHOLD
                    )
                    / (255.0 - XINPUT_GAMEPAD_TRIGGER_THRESHOLD)
                    if state.Gamepad.bLeftTrigger > XINPUT_GAMEPAD_TRIGGER_THRESHOLD
                    else 0.0
                )
                right_trigger = (
                    max(
                        0,
                        state.Gamepad.bRightTrigger - XINPUT_GAMEPAD_TRIGGER_THRESHOLD,
                    )
                    / (255.0 - XINPUT_GAMEPAD_TRIGGER_THRESHOLD)
                    if state.Gamepad.bRightTrigger > XINPUT_GAMEPAD_TRIGGER_THRESHOLD
                    else 0.0
                )

                # Update state
                with self._lock:
                    self.left_thumb_x = thumb_x
                    self.left_trigger = left_trigger
                    self.right_trigger = right_trigger

                    # Combine state for turning and acceleration
                    self.turning = thumb_x
                    self.acceleration = right_trigger - left_trigger

            time.sleep(0.01)  # 10ms polling interval

    def _monitor_keyboard(self):
        """Thread function to monitor keyboard state"""
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        GetAsyncKeyState = user32.GetAsyncKeyState

        while self._running:
            # Get WASD key states
            w_state = GetAsyncKeyState(VK_W)
            a_state = GetAsyncKeyState(VK_A)
            s_state = GetAsyncKeyState(VK_S)
            d_state = GetAsyncKeyState(VK_D)

            # Update state
            with self._lock:
                self.w_pressed = 1 if (w_state & 0x8000) != 0 else 0
                self.a_pressed = 1 if (a_state & 0x8000) != 0 else 0
                self.s_pressed = 1 if (s_state & 0x8000) != 0 else 0
                self.d_pressed = 1 if (d_state & 0x8000) != 0 else 0

                # Combine state for turning and acceleration
                self.turning = self.a_pressed - self.d_pressed
                self.acceleration = self.w_pressed - self.s_pressed

            time.sleep(0.01)  # 10ms polling interval

    def _apply_deadzone(self, value, deadzone):
        """Apply deadzone and normalize value to [-1.0, 1.0] range"""
        if abs(value) < deadzone:
            return 0.0

        # Calculate direction
        sign = 1 if value > 0 else -1

        # Apply deadzone and normalize
        normalized = (abs(value) - deadzone) / (32767.0 - deadzone)
        return sign * min(normalized, 1.0)

    def get_controller_state(self):
        """Get current controller state"""
        with self._lock:
            return {
                "left_thumb_x": self.left_thumb_x,
                "left_trigger": self.left_trigger,
                "right_trigger": self.right_trigger,
            }

    def get_keyboard_state(self):
        """Get current keyboard state"""
        with self._lock:
            return {
                "w": self.w_pressed,
                "a": self.a_pressed,
                "s": self.s_pressed,
                "d": self.d_pressed,
            }

    def get_combined_state(self):
        """Get combined state for turning and acceleration"""
        with self._lock:
            return {
                "turning": self.turning,
                "acceleration": self.acceleration,
            }
