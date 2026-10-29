
import ctypes
import re
import threading
import time
from ctypes import wintypes
from collections import deque

import win32gui

# A&D EJ-02 USB HID identifiers
SCALE_VID = "VID_0DBC"
SCALE_PID = "PID_0005"

WM_INPUT = 0x00FF
RID_INPUT = 0x10000003
RIDI_DEVICENAME = 0x20000007
RIDEV_INPUTSINK = 0x00000100
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104

user32 = ctypes.WinDLL("user32", use_last_error=True)


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [
        ("MakeCode", wintypes.USHORT),
        ("Flags", wintypes.USHORT),
        ("Reserved", wintypes.USHORT),
        ("VKey", wintypes.USHORT),
        ("Message", wintypes.UINT),
        ("ExtraInformation", wintypes.ULONG),
    ]


class RAWINPUT(ctypes.Structure):
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("keyboard", RAWKEYBOARD),
    ]


class EJ02ScaleReader:
    def __init__(self):
        self.latest_weight = None
        self.last_reading_time = None
        self.recent_readings = deque(maxlen=30)
        self.buffer = ""
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(
            target=self._listen, daemon=True
        )
        self.thread.start()
        self.ready.wait(timeout=5)

    def _device_name(self, handle):
        size = wintypes.UINT(0)
        user32.GetRawInputDeviceInfoW(
            handle, RIDI_DEVICENAME, None,
            ctypes.byref(size)
        )
        if not size.value:
            return ""

        buf = ctypes.create_unicode_buffer(size.value)
        result = user32.GetRawInputDeviceInfoW(
            handle, RIDI_DEVICENAME, buf,
            ctypes.byref(size)
        )
        return buf.value.upper() if result != 0xFFFFFFFF else ""

    def _process_key(self, key):
        # Digits from standard keyboard or numeric keypad
        if 48 <= key <= 57:
            self.buffer += chr(key)
        elif 96 <= key <= 105:
            self.buffer += str(key - 96)
        elif key in (110, 190):
            self.buffer += "."
        elif key in (107, 187):
            self.buffer += "+"
        elif key in (109, 189):
            self.buffer += "-"
        elif key == 13:
            value = self.buffer.strip()
            self.buffer = ""

            if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
                with self.lock:
                    weight = float(value)
                    now = time.monotonic()

                    self.latest_weight = weight
                    self.last_reading_time = now
                    self.recent_readings.append((now, weight))

        # Prevent unlimited buffer growth
        if len(self.buffer) > 32:
            self.buffer = ""

    def _window_proc(self, hwnd, msg, wparam, lparam):
        if msg == WM_INPUT:
            size = wintypes.UINT(0)
            user32.GetRawInputData(
                lparam, RID_INPUT, None,
                ctypes.byref(size),
                ctypes.sizeof(RAWINPUTHEADER)
            )

            if not size.value:
                return 0

            buf = ctypes.create_string_buffer(size.value)
            result = user32.GetRawInputData(
                lparam, RID_INPUT, buf,
                ctypes.byref(size),
                ctypes.sizeof(RAWINPUTHEADER)
            )

            if result != 0xFFFFFFFF:
                raw = ctypes.cast(
                    buf, ctypes.POINTER(RAWINPUT)
                ).contents

                device = self._device_name(raw.header.hDevice)

                if SCALE_VID in device and SCALE_PID in device:
                    if raw.keyboard.Message in (
                        WM_KEYDOWN, WM_SYSKEYDOWN
                    ):
                        self._process_key(raw.keyboard.VKey)

            return 0

        return win32gui.DefWindowProc(
            hwnd, msg, wparam, lparam
        )

    def _listen(self):
        try:
            class_name = (
                "EJ02ScaleWindow_" +
                str(threading.get_ident())
            )

            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = self._window_proc
            wc.lpszClassName = class_name

            atom = win32gui.RegisterClass(wc)

            hwnd = win32gui.CreateWindow(
                atom, "EJ02 Listener", 0,
                0, 0, 100, 100,
                0, 0, 0, None
            )

            rid = RAWINPUTDEVICE(
                0x01, 0x06,
                RIDEV_INPUTSINK, hwnd
            )

            if not user32.RegisterRawInputDevices(
                ctypes.byref(rid), 1,
                ctypes.sizeof(RAWINPUTDEVICE)
            ):
                raise ctypes.WinError(
                    ctypes.get_last_error()
                )

            self.ready.set()
            win32gui.PumpMessages()

        except Exception as exc:
            self.error = exc
            self.ready.set()
            print(f"[EJ-02] Listener error: {exc}")

    def read_stable_weight(self, max_age=2.0, window=1.0, tolerance=0.01):
        with self.lock:
            now = time.monotonic()

            if self.last_reading_time is None:
                return None

            if now - self.last_reading_time > max_age:
                return None

            readings = [
                weight
                for timestamp, weight in self.recent_readings
                if now - timestamp <= window
            ]

        if len(readings) < 5:
            return None

        if max(readings) - min(readings) > tolerance:
            return None

        return round(sum(readings) / len(readings), 3)
    
    
    def read_weight_as_value(self):
        with self.lock:
            if self.last_reading_time is None:
                return None

            if time.monotonic() - self.last_reading_time > 2.0:
                return None

            return self.latest_weight

    def read_weight(self):
        value = self.read_weight_as_value()
        return f"{value:.3f} g" if value is not None else None


if __name__ == "__main__":
    import time

    reader = EJ02ScaleReader()

    if reader.error:
        raise reader.error

    print("EJ-02 reader started.")
    print("Place weight on balance and press PRINT.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
