import os
import sys


global base_path

if getattr(sys, 'frozen', False):
    # Running inside PyInstaller .exe
    base_path = sys._MEIPASS
    print(base_path)
    sys.path.insert(0, base_path)
else:
    # Running from source
    base_path = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, base_path)

import tkinter as tk
from tkinter import ttk, font, simpledialog, messagebox
from ctypes import windll
from PIL import Image, ImageTk
from ultralytics import YOLO, SAM, FastSAM
from read_gps.gps_reader import GPSReader
import numpy as np
import torch
import cv2
import csv
import re
import usb.core
import threading
import copy
import hid
import time
from datetime import datetime

import traceback, logging

# ======== For the Icon to appear on the task bar ========
import ctypes
agsolaire_app_id = 'mycompany.myproduct.subproduct.version'  # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(agsolaire_app_id)

from read_weight import scale_reader
from read_weight.ej02_scale_reader import EJ02ScaleReader
from seed_measurement import seed_measurement

# ===== Error logging setup =====
logging.basicConfig(filename="app_error.log", level=logging.ERROR)

def log_exception(exc_type, exc_value, exc_traceback):
    """Log all uncaught exceptions."""
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    logging.error("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

sys.excepthook = log_exception

# GPS: Warmup + cache
gps_reader_global = None
latest_gps_fix = None
gps_lock = threading.Lock()

def gps_warmup_loop(total_seconds=120):
  
    global gps_reader_global, latest_gps_fix

    try:
        gps_reader_global = GPSReader()
        print("[GPS] Warmup started...")

        end = time.time() + total_seconds
        while time.time() < end:
            fix = gps_reader_global.get_fix(max_wait_sec=10)
            if fix:
                with gps_lock:
                    latest_gps_fix = fix
                print("[GPS] Fix acquired during warmup:", fix)
                return

        print("[GPS] Warmup finished but no fix yet.")
    except Exception as e:
        logging.error(f"GPS warmup error: {e}", exc_info=True)

def get_best_gps_fix(fallback_wait_sec=30):
    
    global latest_gps_fix

    # 1) Return cached fix if present
    with gps_lock:
        cached = latest_gps_fix
    if cached:
        return cached

    # 2) Fallback blocking read
    try:
        gps = GPSReader()
        fix = gps.get_fix(max_wait_sec=fallback_wait_sec)
        gps.close()
        if fix:
            with gps_lock:
                latest_gps_fix = fix
        return fix
    except Exception as e:
        logging.error(f"GPS fallback error: {e}", exc_info=True)
        return None

# Create Timestamped Run Folder
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder = os.path.join(os.getcwd(), f"Results\\Run_{run_timestamp}")
os.makedirs(run_folder, exist_ok=True)
print(f"[INFO] Run folder created: {run_folder}")

# Create file for saving results
result_csv_filename = os.path.join(run_folder, f"result_{run_timestamp}.csv")
with open(result_csv_filename, mode='w', newline='') as result_file:
    fieldnames = ["Sample ID", "Average_Length_mm", "Average_Width_mm", "Total_Seeds",
                  "Total_Weight", "Thousand_Kernel_Weight", "Latitude", "Longitude",
                  "GPS_UTC", "GPS_Source"]
    writer = csv.DictWriter(result_file, fieldnames=fieldnames)
    writer.writeheader()
print(f"[INFO] Results saved to {result_csv_filename}")

# Spinning Progress bar class
class Spinner(tk.Canvas):
    def __init__(self, parent, size=60, line_width=6, speed=10, color="#4CAF50"):
        super().__init__(parent, width=size, height=size, bg="white", highlightthickness=0)
        self.size = size
        self.line_width = line_width
        self.speed = speed
        self.color = color
        self.angle = 0
        self.running = False
        self.arc = self.create_arc(5, 5, size-5, size-5, start=self.angle,
                                   extent=90, style="arc", width=line_width, outline=color)

    def start(self):
        self.running = True
        self._rotate()

    def stop(self):
        self.running = False

    def _rotate(self):
        if not self.running:
            return
        self.angle = (self.angle - self.speed) % 360
        self.itemconfig(self.arc, start=self.angle)
        self.after(50, self._rotate)

class CameraApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Window and DPI setup
        try:
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        self.title("AgSolaire App")

        # Start the A&D EJ-02 scale listener
        self.ej02_reader = EJ02ScaleReader()

        if self.ej02_reader.error:
            print(f"[SCALE] EJ-02 listener error: {self.ej02_reader.error}")
        else:
            print("[SCALE] EJ-02 listener started")

        if getattr(sys, 'frozen', False):
            base_path_local = sys._MEIPASS
            self.iconbitmap(os.path.join(base_path_local, 'icons', 'Agsolaire_logo.ico'))
        else:
            base_path_local = os.path.dirname(os.path.dirname(__file__))
            self.iconbitmap(r"C:\Users\manasa_raghavaraju\Projects\Agsolaire\AgSolaire\icons\Agsolaire_logo.ico")

        self.state("zoomed")

        # Start GPS warmup in background
        threading.Thread(target=gps_warmup_loop, daemon=True).start()

        # shared data between screens
        self.captured_image_path = None

        # create containers
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        self.frames = {}
        for F in (CameraScreen, ResultScreen):
            frame = F(parent=container, controller=self)
            self.frames[F] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show_frame(CameraScreen)

    def show_frame(self, page):
        frame = self.frames[page]
        frame.tkraise()

    
    def get_scale_weight(self):
        """
        Read from A&D EJ-02 first.
        Fall back to a compatible serial scale if necessary.
        Returns (weight, unit).
        """

        # First: A&D EJ-123 USB HID scale
        if not self.ej02_reader.error:
            weight = self.ej02_reader.read_stable_weight()

            if weight is not None:
                print(f"[SCALE] A&D EJ-123: {weight:.3f} g")
                return weight, "g"

        # Second: previously supported serial scales
        try:
            # Only attempt serial communication when a matching COM port exists
            device_names = [
                "Prolific PL2303GT USB Serial",
                "Dtech USB Serial Controller",
                "USB-SERIAL CH340",
                "USB Serial Port"
            ]

            import serial.tools.list_ports

            available_ports = serial.tools.list_ports.comports()

            if not any(
                any(name.lower() in port.description.lower() for name in device_names)
                for port in available_ports
            ):
                return None, "g"

            serial_scale = scale_reader.scale_reader()

            if serial_scale._serial is not None:
                weight = serial_scale.read_weight_as_value()

                if weight is not None:
                    print(f"[SCALE] Serial scale: {weight:.3f} g")
                    return weight, "g"

        except Exception as e:
            print(f"[SCALE] Serial scale error: {e}")

        print("[SCALE] No valid weight available.")
        return None, "g"


    
    def is_ej02_connected(self):
        """Check whether the A&D EJ-02 USB interface is connected."""
        try:
            return bool(hid.enumerate(0x0DBC, 0x0005))
        except Exception as e:
            print(f"[SCALE] EJ-02 detection error: {e}")
            return False

    def request_ej02_weight(self):
        """Ask the operator to press PRINT once and capture a new reading."""
        reader = self.ej02_reader

        if reader.error:
            messagebox.showerror("Scale Error", str(reader.error))
            return None

        # Discard any reading received before this sample.
        with reader.lock:
            reader.latest_weight = None
            reader.last_reading_time = None
            reader.buffer = ""

        dialog = tk.Toplevel(self)
        dialog.title("A&D EJ-123 - Weight Measurement")
        dialog.geometry("440x230")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        tk.Label(
            dialog,
            text="Please press PRINT on the weighing scale",
            font=("Arial", 13, "bold"),
            wraplength=390
        ).pack(pady=(25, 10))

        tk.Label(
            dialog,
            text="Wait for the balance to stabilize, then press PRINT once.",
            wraplength=390
        ).pack(pady=5)

        status = tk.Label(
            dialog,
            text="Waiting for weight...",
            font=("Arial", 12),
            fg="blue"
        )
        status.pack(pady=10)

        result = {"weight": None}

        # The EJ-02 acts like a keyboard. Focus this dedicated
        # field so its output does not enter the Sample ID field.
        entry = tk.Entry(dialog, width=25)
        entry.pack()
        entry.focus_set()

        def check_weight():
            if not dialog.winfo_exists():
                return

            # Prefer the reading from our device-specific Raw Input listener.
            with reader.lock:
                value = reader.latest_weight
                received_at = reader.last_reading_time

            if (
                value is not None
                and received_at is not None
                and time.monotonic() - received_at < 5
            ):
                result["weight"] = value
                dialog.destroy()
                return

            dialog.after(100, check_weight)

        def cancel():
            dialog.destroy()

        tk.Button(
            dialog, text="Cancel", command=cancel
        ).pack(pady=10)

        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.after(100, check_weight)

        self.wait_window(dialog)
        return result["weight"]



class CameraScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller

        # Find available cameras
        self.available_cameras = self.find_available_cameras(max_tested=10)
        if not self.available_cameras:
            self.available_cameras = [0]

        self.current_camera_index = self.available_cameras[0]

        # Open initial camera
        self.cap = cv2.VideoCapture(self.current_camera_index)

        self.label = tk.Label(self)
        self.label.pack(pady=20)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        ttk.Style().configure('My.TButton', font=('Helvetica', 11, "bold"), padding=(10, 5))

        # Capture button
        capture_icon_path = os.path.join(base_path, 'icons', 'camera.png')
        self.capture_icon = ImageTk.PhotoImage(Image.open(capture_icon_path))
        capture_btn = ttk.Button(btn_frame, text="Capture Image", command=self.capture_image,
                                 style='My.TButton', image=self.capture_icon, compound="left")
        capture_btn.grid(row=0, column=0, padx=5)

        # Display labels like "Camera 0", "Camera 1", ...
        camera_labels = [f"Camera {idx}" for idx in self.available_cameras]
        self.selected_camera = tk.StringVar(value=camera_labels[0])

        self.camera_combo = ttk.Combobox(btn_frame, textvariable=self.selected_camera,
                                         values=camera_labels, state="readonly", width=12)
        self.camera_combo.grid(row=0, column=1, padx=5)

        # When selection changes, switch camera
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_change)

        # Quit button
        quit_icon_path = os.path.join(base_path, 'icons', 'close.png')
        self.quit_icon = ImageTk.PhotoImage(Image.open(quit_icon_path))
        quit_btn = ttk.Button(btn_frame, text="Quit", command=self.quit_app,
                              style='My.TButton', image=self.quit_icon, compound="left")
        quit_btn.grid(row=0, column=2, padx=5)

        self.update_frame()

    def on_camera_change(self, event=None):
        label = self.selected_camera.get()
        idx_str = label.split()[-1]
        try:
            new_index = int(idx_str)
        except ValueError:
            return

        if new_index == self.current_camera_index:
            return

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(new_index)
        if self.cap.isOpened():
            self.current_camera_index = new_index
            print(f"[INFO] Switched to camera index {new_index}")
        else:
            print(f"[WARN] Unable to open camera index {new_index}")
            self.cap = cv2.VideoCapture(self.current_camera_index)

    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                frame = cv2.resize(frame, (800, 600))
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)

        self.after(10, self.update_frame)

    def capture_image(self):
        if hasattr(self, 'current_frame'):
            filename = "captured_image.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.controller.captured_image_path = filename
            self.controller.show_frame(ResultScreen)

    def quit_app(self):
        if self.cap is not None:
            self.cap.release()
        self.controller.destroy()

    def find_available_cameras(self, max_tested=10):
        indices = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                indices.append(i)
                cap.release()
        print(f"[INFO] Available cameras: {indices}")
        return indices

class ResultScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        image_frame = tk.Frame(self)
        image_frame.pack(side="left", anchor="nw", pady=20, padx=20)

        self.controls_frame = tk.Frame(self)
        self.controls_frame.pack(side="left", anchor="nw")

        self.image_label = tk.Label(image_frame)
        self.image_label.pack()

        btn_frame = tk.Frame(self.controls_frame)
        btn_frame.pack()

        ttk.Style().configure('My.TButton', font=('Helvetica', 11, "bold"), padding=(10, 5))

        back_icon_path = os.path.join(base_path, 'icons', 'back.png')
        self.back_icon = ImageTk.PhotoImage(Image.open(back_icon_path))
        back_btn = ttk.Button(btn_frame, text="Back to Camera", command=self.go_back,
                              style='My.TButton', image=self.back_icon, compound="left")
        back_btn.pack(pady=20, padx=20, side="left")

        run_model_icon_path = os.path.join(base_path, 'icons', 'scanning.png')
        self.run_model_icon = ImageTk.PhotoImage(Image.open(run_model_icon_path))
        infer_btn = ttk.Button(btn_frame, text="Send for Inference",
                               command=self.run_inference_button_click,
                               style='My.TButton', image=self.run_model_icon, compound="left")
        infer_btn.pack(pady=20, padx=10)

        results_label = tk.Frame(self.controls_frame)
        results_label.pack()

        self.spinner = Spinner(self.controls_frame, size=60, color="#4CAF50", speed=20)

        self.inference_label = tk.Label(results_label, text="", font=("Arial", 10, "bold"))
        self.inference_label.pack(pady=10)

        self.count_label = tk.Label(results_label, text="", font=("Arial", 10, "bold"))
        self.count_label.pack(pady=10)

        self.weight_label = tk.Label(results_label, text="", font=("Arial", 10, "bold"))
        self.weight_label.pack(pady=10)

        self.TKW_label = tk.Label(results_label, text="", font=("Arial", 10, "bold"))
        self.TKW_label.pack(pady=10)

        self.mm_to_pxl_label = tk.Label(results_label, text="", font=("Arial", 10, "bold"))
        self.mm_to_pxl_label.pack(pady=10)

        save_icon_path = os.path.join(base_path, 'icons', 'diskette.png')
        self.save_icon = ImageTk.PhotoImage(Image.open(save_icon_path))
        self.save_results_btn = ttk.Button(self.controls_frame, text="Save Results",
                                           command=self.save_results,
                                           style='My.TButton', image=self.save_icon, compound="left")

        self.saving_label = tk.Label(self.controls_frame, text="", font=("Arial", 12))
        self.saving_label.pack(pady=20)

                # ===== Current Run Results Table =====
        self.results_table_frame = tk.LabelFrame(
            self.controls_frame,
            text="Current Run Results",
            font=("Arial", 11, "bold"),
            padx=10,
            pady=10
        )
        # self.results_table_frame.pack(
        #     fill="both",
        #     expand=True,
        #     padx=10,
        #     pady=10
        # )

        columns = (
            "sample_id",
            "avg_length",
            "avg_width",
            "seed_count",
            "weight",
            "tkw",
            "latitude",
            "longitude",
            "gps_utc",
            "gps_source"
        )

        self.results_table = ttk.Treeview(
            self.results_table_frame,
            columns=columns,
            show="headings",
            height=8
        )

        self.results_scrollbar = ttk.Scrollbar(
            self.results_table_frame,
            orient="vertical",
            command=self.results_table.yview
        )

        self.results_h_scrollbar = ttk.Scrollbar(
            self.results_table_frame,
            orient="horizontal",
            command=self.results_table.xview
        )

        self.results_table.configure(
            yscrollcommand=self.results_scrollbar.set,
            xscrollcommand=self.results_h_scrollbar.set
        )

        self.results_table.heading("sample_id", text="Sample ID")
        self.results_table.heading("avg_length", text="Avg Length (mm)")
        self.results_table.heading("avg_width", text="Avg Width (mm)")
        self.results_table.heading("seed_count", text="Seeds")
        self.results_table.heading("weight", text="Weight")
        self.results_table.heading("tkw", text="TKW")
        self.results_table.heading("latitude", text="Latitude")
        self.results_table.heading("longitude", text="Longitude")
        self.results_table.heading("gps_utc", text="GPS UTC")
        self.results_table.heading("gps_source", text="GPS Source")


        self.results_table.column("sample_id", width=100, anchor="center")
        self.results_table.column("avg_length", width=110, anchor="center")
        self.results_table.column("avg_width", width=110, anchor="center")
        self.results_table.column("seed_count", width=70, anchor="center")
        self.results_table.column("weight", width=90, anchor="center")
        self.results_table.column("tkw", width=90, anchor="center")
        self.results_table.column("latitude", width=100, anchor="center")
        self.results_table.column("longitude", width=100, anchor="center")
        self.results_table.column("gps_utc", width=120, anchor="center")
        self.results_table.column("gps_source", width=90, anchor="center")


        self.results_scrollbar.pack(side="right", fill="y")
        self.results_h_scrollbar.pack(side="bottom", fill="x")
        self.results_table.pack(side="left", fill="both", expand=True)

        # Button to open the permanent CSV results file
        self.open_results_btn = tk.Button(
            self.controls_frame,
            text="Open Results File",
            command=self.open_results_file,
            font=("Arial", 10, "bold")
        )
        
        self.image_size = (800, 600)

    def tkraise(self, *args, **kwargs):
        super().tkraise(*args, **kwargs)
        self.show_captured_image()

    def show_captured_image(self):
        path = self.controller.captured_image_path
        if path and os.path.exists(path):
            img = Image.open(path).resize(self.image_size)
            imgtk = ImageTk.PhotoImage(img)
            self.image_label.imgtk = imgtk
            self.image_label.config(image=imgtk)
        else:
            self.image_label.config(text="No image captured yet.")

    def go_back(self):
        self.controller.show_frame(CameraScreen)

        self.count_label.config(text="")
        self.weight_label.config(text="")
        self.TKW_label.config(text="")
        self.mm_to_pxl_label.config(text="")
        self.saving_label.config(text="")
        self.inference_label.config(text="")

        self.spinner.pack_forget()
        self.save_results_btn.pack_forget()

        torch.cuda.empty_cache()

    
    def run_inference_button_click(self):

        self.pending_weight = None
        self.pending_weight_unit = "g"
        self.use_ej02_for_inference = False

        # A&D EJ-02: request one PRINT reading.
        if self.controller.is_ej02_connected():

            self.use_ej02_for_inference = True

            weight = self.controller.request_ej02_weight()

            if weight is None:
                messagebox.showwarning(
                    "Weight Not Received",
                    "No weight was captured. Inference was cancelled."
                )
                return

            self.pending_weight = weight
            print(f"[SCALE] A&D EJ-123: {weight:.3f} g")

        # Serial scales do not require the PRINT dialog.
        self.inference_label.config(
            text="Model Inference running ... !!!!!"
        )
        self.spinner.pack(pady=10)
        self.spinner.start()

        threading.Thread(
            target=self.run_inference,
            daemon=True
        ).start()


    def run_inference(self):
        if getattr(sys, 'frozen', False):
            base_path_local = sys._MEIPASS
            self.yolo_model = YOLO(os.path.join(base_path_local, "model", "yolo_detection_model.pt"))
            self.sam_model = SAM(os.path.join(base_path_local, "model", "mobile_sam.pt"))
        else:
            base_path_local = os.path.dirname(os.path.dirname(__file__))
            self.yolo_model = YOLO(os.path.join(base_path_local, "model", "yolo_detection_model.pt"))
            self.sam_model = SAM(os.path.join(base_path_local, "model", "mobile_sam.pt"))

        self.image = cv2.imread(self.controller.captured_image_path)

        
        # Read weight from A&D EJ-123 or previously supported serial scales
        if self.use_ej02_for_inference:
            # A&D EJ-123: use weight captured after pressing PRINT
            value = self.pending_weight
            unit = self.pending_weight_unit

        else:
            # Previously supported serial scales: automatic reading
            value, unit = self.controller.get_scale_weight()

        self.seed_weight = value

        self.seed_weight_wt_unit = (
            f"{value:.3f} {unit}"
            if value is not None
            else "Unavailable"
        )

        print(f"[SCALE] Weight for inference: {self.seed_weight_wt_unit}")


        pixel_to_conversion = seed_measurement(self.image)
        scale_calculation = pixel_to_conversion.calculate_length_width_in_mm()

        self.mm_per_pixel = None
        if scale_calculation and "mm_per_pixel" in scale_calculation:
            self.mm_per_pixel = scale_calculation["mm_per_pixel"]

        results = self.yolo_model.predict(source=self.image, conf=0.5)
        boxes = results[0].boxes.xywh.cpu().numpy()
        self.seed_count = len(boxes)

        self.TKW = (
            (self.seed_weight / self.seed_count) * 1000
            if self.seed_weight is not None and self.seed_count > 0
            else None
        )

        sam_results = self.sam_model.predict(self.image, points=boxes[:, :2])

        overlay = self.image.copy()
        overlay_with_seed_id = self.image.copy()
        seed_id = 0
        seed_data = []

        for r in sam_results:
            masks = r.masks.data.cpu().numpy()
            for mask in masks:
                seed_id += 1
                mask = (mask > 0.5).astype(np.uint8)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                cnt = max(contours, key=cv2.contourArea)
                if len(cnt) < 5:
                    continue

                ellipse = cv2.fitEllipse(cnt)
                (cx, cy), (MA, ma), angle = ellipse

                length_px = max(MA, ma)
                width_px = min(MA, ma)

                if self.mm_per_pixel:
                    length_mm = length_px * self.mm_per_pixel
                    width_mm = width_px * self.mm_per_pixel
                else:
                    length_mm = None
                    width_mm = None

                cv2.ellipse(overlay, ellipse, (0, 255, 0), 4)
                cv2.circle(overlay, (int(cx), int(cy)), 4, (0, 255, 0), -1)

                seed_data.append({
                    "Seed_ID": seed_id,
                    "Length_mm": round(length_mm, 3) if length_mm is not None else None,
                    "Width_mm": round(width_mm, 3) if width_mm is not None else None
                })

                seed_label = f"id:{seed_id}"
                cv2.putText(overlay_with_seed_id, seed_label, (int(cx) + 2, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out = cv2.addWeighted(self.image, 0.5, overlay, 0.5, 0)
        self.out_with_seed_id = cv2.addWeighted(out, 0.7, overlay_with_seed_id, 0.3, 0.3)

        out_resized = cv2.resize(out, self.image_size)
        self.out_with_seed_id = cv2.resize(self.out_with_seed_id, self.image_size)

        valid_lengths = [s["Length_mm"] for s in seed_data if s["Length_mm"] is not None]
        valid_widths = [s["Width_mm"] for s in seed_data if s["Width_mm"] is not None]

        self.total_length_mm = round(sum(valid_lengths), 3) if valid_lengths else 0.0
        self.total_width_mm = round(sum(valid_widths), 3) if valid_widths else 0.0

        self.avg_length = round(self.total_length_mm / len(valid_lengths), 3) if valid_lengths else 0.0
        self.avg_width = round(self.total_width_mm / len(valid_widths), 3) if valid_widths else 0.0

        inference_image = cv2.cvtColor(out_resized, cv2.COLOR_BGR2RGB)
        inference_image = Image.fromarray(inference_image)

        self.spinner.pack_forget()
        self.inference_label.config(text="Inference Completed !!!")

        imgtk = ImageTk.PhotoImage(inference_image)
        self.image_label.imgtk = imgtk
        self.image_label.config(image=imgtk)

        self.count_label.config(text=f"seed count: {self.seed_count}")
        self.weight_label.config(text=f"Weight: {self.seed_weight_wt_unit}")
        self.TKW_label.config(
            text=(
                f"Thousand Kernel Weight: {self.TKW:.3f} g"
                if self.TKW is not None
                else "Thousand Kernel Weight: Unavailable"
            )
        )
        self.mm_to_pxl_label.config(text=f"mm/pxl: {self.mm_per_pixel:.3f}" if self.mm_per_pixel else "mm/pxl: None")

        self.save_results_btn.pack(pady=20, padx=20, side="top")

        del sam_results
        del results
        del self.sam_model
        del self.yolo_model

        print("[INFO] Inference completed successfully.")

    def save_results(self):
        self.sample_ID = simpledialog.askstring("Barcode Entry", "Sample ID:")

        gps_lat = ""
        gps_lon = ""
        gps_utc = ""
        gps_source = ""

        # NEW: Prefer cached warmup fix; fallback to blocking read
        fix = get_best_gps_fix(fallback_wait_sec=30)
        if fix:
            gps_lat = fix.get("latitude", "")
            gps_lon = fix.get("longitude", "")
            gps_utc = fix.get("utc", "")
            gps_source = fix.get("source", "")

        with open(result_csv_filename, mode='a', newline='') as result_file:
            fieldnames = ["Sample ID", "Average_Length_mm", "Average_Width_mm", "Total_Seeds",
                          "Total_Weight", "Thousand_Kernel_Weight", "Latitude", "Longitude",
                          "GPS_UTC", "GPS_Source"]
            writer = csv.DictWriter(result_file, fieldnames=fieldnames)

            writer.writerow({
                "Sample ID": self.sample_ID,
                "Average_Length_mm": self.avg_length,
                "Average_Width_mm": self.avg_width,
                "Total_Seeds": self.seed_count,
                "Total_Weight": (
                    round(self.seed_weight, 3)
                    if self.seed_weight is not None else ""
                ),
                "Thousand_Kernel_Weight": (
                    round(self.TKW, 3)
                    if self.TKW is not None else ""
                ),
                "Latitude": gps_lat,
                "Longitude": gps_lon,
                "GPS_UTC": gps_utc,
                "GPS_Source": gps_source,
            })

        print(f"[INFO] Results saved to {result_csv_filename}")

        masked_image_filename = os.path.join(run_folder, f"masked_image_sample_{self.sample_ID}.jpg")
        cv2.imwrite(masked_image_filename, self.out_with_seed_id)

        self.save_results_btn.pack_forget()
        self.saving_label.config(text="Results saved !!!!", font=("Arial", 10, "bold"))

        # Show Current Run Results only after a sample has been saved
        if not self.results_table_frame.winfo_ismapped():
            self.results_table_frame.pack(
                fill="both",
                expand=True,
                padx=10,
                pady=10
            )

        # Show Open Results File button after the first successful save
        if not self.open_results_btn.winfo_ismapped():
            self.open_results_btn.pack(pady=(0, 10))

        # Add the successfully saved sample to the Current Run Results table
        new_row = self.results_table.insert(
            "",
            "end",
            values=(
                self.sample_ID,
                self.avg_length,
                self.avg_width,
                self.seed_count,
                round(self.seed_weight, 3) if self.seed_weight is not None else "",
                round(self.TKW, 3) if self.TKW is not None else "",
                gps_lat,
                gps_lon,
                gps_utc,
                gps_source
            )
        )

        self.results_table.after(
            100,
            lambda row=new_row: self.results_table.see(row)
        )

    def open_results_file(self):
        try:
            if os.path.exists(result_csv_filename):
                os.startfile(result_csv_filename)
            else:
                messagebox.showerror(
                    "File Not Found",
                    "The results file could not be found."
                )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Could not open the results file:\n{e}"
            )

if __name__ == "__main__":
    import time  # needed for gps_warmup_loop
    app = CameraApp()
    try:
        app.mainloop()
    except Exception:
        traceback.print_exc()