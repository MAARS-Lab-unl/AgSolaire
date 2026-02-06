import time
import serial
import serial.tools.list_ports
import pynmea2
from serial.serialutil import SerialException


class GPSReader:
    def __init__(self, port=None, baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

        # If port not provided, auto-detect using Device Manager description
        if self.port is None:
            device_names = [
                "u-blox 7 GPS/GNSS Receiver",   # your exact device description
                "u-blox",                       # fallback
                "GPS/GNSS",                     # fallback
            ]
            self.port = self.find_device(device_names)

    # def find_device(self, device_list):
    #     ports = serial.tools.list_ports.comports()
    #     for p in ports:
    #         for device_name in device_list:
    #             if device_name.lower() in p.description.lower():
    #                 print(f"[GPS] Found {p.description} at {p.device}")
    #                 return p.device
    #     print("[GPS] No matching GPS device found.")
    #     return None

    def find_device(self, device_list):
        ports = list(serial.tools.list_ports.comports())

        # 1) Try description match first (works on your laptop)
        for p in ports:
            desc = (p.description or "").lower()
            for device_name in device_list:
                if device_name.lower() in desc:
                    print(f"[GPS] Found by name: {p.description} at {p.device}")
                    return p.device

        # 2) Fallback: probe ports for NMEA output (works on tablet)
        # Skip obvious scale adapters
        skip_keywords = ["ch340", "prolific", "dtech", "pl2303"]
        candidate_ports = []
        for p in ports:
            desc = (p.description or "").lower()
            if any(k in desc for k in skip_keywords):
                continue
            candidate_ports.append(p.device)

        # If everything got skipped, probe all ports
        if not candidate_ports:
            candidate_ports = [p.device for p in ports]

        for dev in candidate_ports:
            try:
                with serial.Serial(dev, self.baudrate, timeout=1) as s:
                    start = time.time()
                    while time.time() - start < 2:
                        line = s.readline().decode("ascii", errors="ignore").strip()
                        if line.startswith(("$GP", "$GN", "$GL", "$GA", "$GB")):
                            print(f"[GPS] Found by NMEA probe: {dev}")
                            return dev
            except Exception:
                continue

        print("[GPS] No matching GPS device found.")
        return None

    def open(self):
        if not self.port:
            raise RuntimeError("GPS COM port not found. Plug in GPS receiver and try again.")

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        except FileNotFoundError:
            raise RuntimeError(f"GPS serial port '{self.port}' not found.")
        except SerialException as e:
            raise RuntimeError(f"GPS serial port error on '{self.port}': {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def get_fix(self, max_wait_sec=15):
        """
        Returns dict with lat/lon/utc or None if no fix
        """
        if not self.ser or not self.ser.is_open:
            self.open()

        end_time = time.time() + max_wait_sec

        while time.time() < end_time:
            try:
                line = self.ser.readline().decode("ascii", errors="ignore").strip()
                if not line.startswith("$"):
                    continue

                msg = pynmea2.parse(line)

                # RMC has valid/invalid status
                if msg.sentence_type == "RMC" and getattr(msg, "status", "") == "A":
                    return {
                        "latitude": msg.latitude,
                        "longitude": msg.longitude,
                        "utc": str(msg.timestamp),
                        "source": "RMC"
                    }

                # GGA gives fix quality and satellites
                if msg.sentence_type == "GGA" and int(getattr(msg, "gps_qual", 0) or 0) > 0:
                    return {
                        "latitude": msg.latitude,
                        "longitude": msg.longitude,
                        "utc": str(msg.timestamp),
                        "fix_quality": msg.gps_qual,
                        "satellites": msg.num_sats,
                        "source": "GGA"
                    }

            except Exception:
                continue

        return None