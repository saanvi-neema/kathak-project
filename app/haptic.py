"""
Haptic glove interface — sends buzz commands to Arduino over serial.

Finger index mapping (matches hand landmark order):
  0 = thumb, 1 = index, 2 = middle, 3 = ring, 4 = pinky

Arduino expects two bytes: b'F' + finger_index_byte
"""

import serial
import threading
import time

# Change this to match your COM port (check Device Manager)
SERIAL_PORT = "COM5"
BAUD_RATE = 9600
BUZZ_DURATION_MS = 300  # how long Arduino holds the motor on

COOLDOWN_SECS = 5.0

_serial_lock = threading.Lock()
_ser = None
_last_buzz_time = 0.0


def _get_connection():
    global _ser
    if _ser is None or not _ser.is_open:
        _ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # wait for Arduino reset after connection
    return _ser


def can_buzz() -> bool:
    """True if the cooldown since the last buzz has elapsed."""
    return time.time() - _last_buzz_time >= COOLDOWN_SECS


def buzz_finger(finger: int) -> bool:
    """Buzz motor for one finger (0=thumb … 4=pinky). Returns True on success."""
    global _last_buzz_time
    if not (0 <= finger <= 4):
        return False
    try:
        with _serial_lock:
            ser = _get_connection()
            ser.write(bytes([ord('F'), finger]))
        _last_buzz_time = time.time()
        return True
    except Exception as e:
        print(f"[haptic] serial error: {e}")
        return False


def buzz_fingers(fingers: list[int]) -> None:
    """Buzz multiple fingers with a short gap between each."""
    for f in fingers:
        buzz_finger(f)
        time.sleep(0.05)


def close():
    global _ser
    with _serial_lock:
        if _ser and _ser.is_open:
            _ser.close()
            _ser = None
