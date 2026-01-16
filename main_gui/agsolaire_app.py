import os
import sys

# ======== adding path for other modules ========
global base_path

if getattr(sys, 'frozen', False):
    # Running inside PyInstaller .exe
    base_path = sys._MEIPASS
    # sys.path.insert(0,os.path.join(base_path, 'read_weight'))
    print(base_path)
    sys.path.insert(0,base_path)
else:
    # Running from source — your local path
    base_path = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0,base_path)

import tkinter as tk
from tkinter import ttk,font,simpledialog
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
from datetime import datetime

import traceback, logging


# ======== For the Icon to appear on the task bar ========
import ctypes
agsolaire_app_id = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(agsolaire_app_id)


# module_path = r"C:\Users\hmwunguzi2\Documents\AgSolaire-main"
# sys.path.insert(0,module_path)

from read_weight import scale_reader
from seed_measurement import seed_measurement


# ===== Error logging setup =====
logging.basicConfig(filename="app_error.log", level=logging.ERROR)

def log_exception(exc_type, exc_value, exc_traceback):
    """Log all uncaught exceptions."""
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    logging.error("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

sys.excepthook = log_exception

# creating folder and file for saving results

# ======== Create Timestamped Run Folder ========
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder = os.path.join(os.getcwd(), f"Results\Run_{run_timestamp}")
os.makedirs(run_folder, exist_ok=True)
print(f"[INFO] Run folder created: {run_folder}")

# ======== Create file for saving results ========
result_csv_filename = os.path.join(run_folder, f"result_{run_timestamp}.csv")
with open(result_csv_filename, mode='w', newline='') as result_file:
    fieldnames = ["Sample ID","Average_Length_mm", "Average_Width_mm", "Total_Seeds", "Total_Weight", "Thousand_Kernel_Weight", "Latitude", "Longitude", "GPS_UTC", "GPS_Source"]
    # fieldnames = ["Sample ID", "Average_Length_mm", "Average_Width_mm", "Total_Length_mm", "Total_Width_mm", "Total_Seeds", "Total_Weight", "Thousand_Kernel_Weight"]
    writer = csv.DictWriter(result_file, fieldnames=fieldnames)
    writer.writeheader()
print(f"[INFO] Results saved to {result_csv_filename}")


#Spinning Progress bar class
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

        # ============ Window and DPI setup ============
        try:
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        
        self.title("AgSolaire App")

        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
            self.iconbitmap(os.path.join(base_path,'icons','Agsolaire_logo.ico'))
        else:
            base_path = os.path.dirname(os.path.dirname(__file__))
            self.iconbitmap(r"C:\Users\manasa_raghavaraju\Projects\Agsolaire\AgSolaire\icons\Agsolaire_logo.ico")

        # self.iconphoto(True,tk.PhotoImage(file = r"C:\Users\hmwunguzi2\Documents\AgSolaire-main\Logo.png"))
        # self.geometry("800x600")
        # self.attributes('-fullscreen', True)
        self.state("zoomed")
       
        # ============ Layout setup ============
        # self.rowconfigure(0, weight=1)
        # self.columnconfigure(0, weight=1)

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
        capture_btn = ttk.Button(btn_frame, text="Capture Image", command=self.capture_image, style='My.TButton', image=self.capture_icon, compound="left")
        capture_btn.grid(row=0, column=0, padx=5)

        # Display labels like "Camera 0", "Camera 1", ...
        camera_labels = [f"Camera {idx}" for idx in self.available_cameras]
        self.selected_camera = tk.StringVar(value=camera_labels[0])

        self.camera_combo = ttk.Combobox(btn_frame, textvariable=self.selected_camera, values=camera_labels, state="readonly", width=12)
        self.camera_combo.grid(row=0, column=1, padx=5)

        # When selection changes, switch camera
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_change)

        # Quit button
        quit_icon_path = os.path.join(base_path, 'icons', 'close.png')
        self.quit_icon = ImageTk.PhotoImage(Image.open(quit_icon_path))
        quit_btn = ttk.Button(btn_frame, text="Quit", command=self.quit_app, style='My.TButton', image=self.quit_icon, compound="left")
        quit_btn.grid(row=0, column=2, padx=5)

        self.update_frame()

    # Camera switching logic
    def on_camera_change(self, event=None):
        """Called when user picks a different camera from the dropdown."""
        label = self.selected_camera.get()         
        idx_str = label.split()[-1]     
        try:
            new_index = int(idx_str)
        except ValueError:
            return

        if new_index == self.current_camera_index:
            return

        # Release current camera and open the new one
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(new_index)
        if self.cap.isOpened():
            self.current_camera_index = new_index
            print(f"[INFO] Switched to camera index {new_index}")
        else:
            print(f"[WARN] Unable to open camera index {new_index}")
            # trying to reopen old one
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
        
        # GUI component initialization
        image_frame = tk.Frame(self)
        image_frame.pack(side="left", anchor="nw", pady=20, padx=20)

        self.controls_frame = tk.Frame(self)
        self.controls_frame.pack(side="left",  anchor="nw")

        self.image_label = tk.Label(image_frame)
        self.image_label.pack()

        btn_frame = tk.Frame(self.controls_frame)
        btn_frame.pack()


        ttk.Style().configure('My.TButton', 
        font=('Helvetica', 11,"bold"),  # Increase font size
        padding=(10, 5))  
        
        back_icon_path = os.path.join(base_path,'icons','back.png')
        self.back_icon = ImageTk.PhotoImage(Image.open(back_icon_path))
        back_btn = ttk.Button(btn_frame, text="Back to Camera", command=self.go_back, style='My.TButton',image=self.back_icon, compound="left")
        back_btn.pack( pady = 20, padx=20, side ="left")

        run_model_icon_path = os.path.join(base_path,'icons','scanning.png')
        self.run_model_icon = ImageTk.PhotoImage(Image.open(run_model_icon_path))
        infer_btn = ttk.Button(btn_frame, text="Send for Inference", command=self.run_inference_button_click, style='My.TButton', image=self.run_model_icon, compound="left")
        infer_btn.pack(pady = 20, padx=10)

        results_label = tk.Frame(self.controls_frame)
        results_label.pack()

        self.spinner = Spinner(self.controls_frame, size=60, color="#4CAF50", speed=20)

        self.inference_label = tk.Label(results_label, text="", font=("Arial", 10,"bold"))
        self.inference_label.pack(pady=10)
        
        self.count_label = tk.Label(results_label, text="", font=("Arial", 10,"bold"))
        self.count_label.pack(pady=10)

        self.weight_label = tk.Label(results_label, text="", font=("Arial", 10,"bold"))
        self.weight_label.pack(pady=10)

        self.TKW_label = tk.Label(results_label, text="", font=("Arial", 10,"bold"))
        self.TKW_label.pack(pady=10)
        
        self.mm_to_pxl_label = tk.Label(results_label, text="", font=("Arial", 10,"bold"))
        self.mm_to_pxl_label.pack(pady=10)

        save_icon_path = os.path.join(base_path,'icons','diskette.png')
        self.save_icon = ImageTk.PhotoImage(Image.open(save_icon_path))
        self.save_results_btn = ttk.Button(self.controls_frame, text="Save Results", command=self.save_results, style='My.TButton', image = self.save_icon, compound="left")

        self.saving_label = tk.Label(self.controls_frame, text="", font=("Arial", 12))
        self.saving_label.pack(pady=20)
        
        self.image_size = (800,600)


    def tkraise(self, *args, **kwargs):
        """Overridden to refresh image each time screen is shown"""
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

        #label
        self.inference_label.config(text="Model Inference running ... !!!!!")

        #create a progress bar and start it
        self.spinner.pack(pady=10)
        self.spinner.start()

        #start a thread for running inference
        threading.Thread(target=self.run_inference,daemon=True).start()


    def run_inference(self):

        # Load models
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
            self.yolo_model = YOLO(os.path.join(base_path, "model", "yolo_detection_model.pt"))
            self.sam_model  = SAM(os.path.join(base_path, "model", "mobile_sam.pt"))
        else:
            base_path = os.path.dirname(os.path.dirname(__file__))
            self.yolo_model = YOLO(os.path.join(base_path, "model", "yolo_detection_model.pt"))
            self.sam_model  = SAM(os.path.join(base_path, "model", "mobile_sam.pt"))

        # Read the captured image once
        self.image = cv2.imread(self.controller.captured_image_path)

        # Weight Measurement
        weight_reading = scale_reader.scale_reader()
        raw_line = weight_reading.read_weight()
        value    = weight_reading.read_weight_as_value()
        self.seed_weight = value if value is not None else 0.0

        unit = ""
        if raw_line:
            m = re.search(r'([a-zA-Z]+)\s*$', raw_line)
            if m:
                unit = m.group(1)
        if not unit:
            unit = "g"

        self.seed_weight_wt_unit = f"{self.seed_weight:.3f} {unit}"

        # mm per pixel scale calculation on the SAME image
        pixel_to_conversion = seed_measurement(self.image)
        scale_calculation   = pixel_to_conversion.calculate_length_width_in_mm()

        self.mm_per_pixel = None
        if scale_calculation and "mm_per_pixel" in scale_calculation:
            self.mm_per_pixel = scale_calculation["mm_per_pixel"]

        #YOLO Detection on the SAME image
        results = self.yolo_model.predict(source=self.image, conf=0.7)
        boxes   = results[0].boxes.xywh.cpu().numpy()
        self.seed_count = len(boxes)

        # Thousand Kernel Weight
        self.TKW = (self.seed_weight / self.seed_count) * 1000 if self.seed_count > 0 else 0

        # SAM Segmentation
        sam_results = self.sam_model.predict(self.image, points=boxes[:, :2])

        overlay              = self.image.copy()
        overlay_with_seed_id = self.image.copy()
        seed_id   = 0
        seed_data = []

        for r in sam_results:
            masks = r.masks.data.cpu().numpy()
            for mask in masks:
                seed_id += 1
                mask = (mask > 0.5).astype(np.uint8)

                # Find contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                cnt = max(contours, key=cv2.contourArea)
                if len(cnt) < 5:
                    continue

                ellipse = cv2.fitEllipse(cnt)   # (center, (MA, ma), angle)
                (cx, cy), (MA, ma), angle = ellipse

                # Major = length, Minor = width
                length_px = max(MA, ma)
                width_px  = min(MA, ma)

                # Convert to mm using mm_per_pixel
                if self.mm_per_pixel:
                    length_mm = length_px * self.mm_per_pixel
                    width_mm  = width_px  * self.mm_per_pixel
                else:
                    length_mm = None
                    width_mm  = None

                cv2.ellipse(overlay, ellipse, (0, 255, 0), 4)
                cv2.circle(overlay, (int(cx), int(cy)), 4, (0, 255, 0), -1)

                seed_data.append({
                    "Seed_ID": seed_id,
                    "Length_mm": round(length_mm, 3) if length_mm is not None else None,
                    "Width_mm":  round(width_mm,  3) if width_mm  is not None else None
                })

                seed_label = f"id:{seed_id}"
                cv2.putText(
                    overlay_with_seed_id,
                    seed_label,
                    (int(cx) + 2, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        # Create overlay images for display (now we resize ONLY for GUI)
        out = cv2.addWeighted(self.image, 0.5, overlay, 0.5, 0)
        self.out_with_seed_id = cv2.addWeighted(out, 0.7, overlay_with_seed_id, 0.3, 0.3)

        out_resized = cv2.resize(out, self.image_size)
        self.out_with_seed_id = cv2.resize(self.out_with_seed_id, self.image_size)

        # Compute totals and averages
        valid_lengths = [s["Length_mm"] for s in seed_data if s["Length_mm"] is not None]
        valid_widths  = [s["Width_mm"]  for s in seed_data if s["Width_mm"]  is not None]

        self.total_length_mm = round(sum(valid_lengths), 3) if valid_lengths else 0.0
        self.total_width_mm  = round(sum(valid_widths),  3) if valid_widths  else 0.0

        self.avg_length = round(self.total_length_mm / len(valid_lengths), 3) if valid_lengths else 0.0
        self.avg_width  = round(self.total_width_mm  / len(valid_widths),  3) if valid_widths  else 0.0

        logging.info(
            f"Seeds={self.seed_count}, len(valid_lengths)={len(valid_lengths)}, "
            f"total_length_mm={self.total_length_mm}, avg_length_mm={self.avg_length}, "
            f"total_width_mm={self.total_width_mm}, avg_width_mm={self.avg_width}, "
            f"mm_per_pixel={self.mm_per_pixel}"
        )

        # 10) Display overlay on GUI
        inference_image = cv2.cvtColor(out_resized, cv2.COLOR_BGR2RGB)
        inference_image = Image.fromarray(inference_image)

        self.spinner.pack_forget()
        self.inference_label.config(text="Inference Completed !!!")

        imgtk = ImageTk.PhotoImage(inference_image)
        self.image_label.imgtk = imgtk
        self.image_label.config(image=imgtk)

        seed_count_string = f"seed count: {self.seed_count}"
        weight_string     = f"Weight: {self.seed_weight_wt_unit}"
        TKW_string        = f"Thousand Kernel Weight: {self.TKW}"
        mm_to_pxl_string  = (
            f"mm/pxl: {self.mm_per_pixel:.3f}" if self.mm_per_pixel else "mm/pxl: None"
        )

        self.count_label.config(text=seed_count_string)
        self.weight_label.config(text=weight_string)
        self.TKW_label.config(text=TKW_string)
        self.mm_to_pxl_label.config(text=mm_to_pxl_string)

        self.save_results_btn.pack(pady=20, padx=20, side="top")

        # Cleanup
        del sam_results
        del results
        del self.sam_model
        del self.yolo_model

        print("[INFO] Inference completed successfully.")
    
    def save_results(self):
        self.sample_ID = simpledialog.askstring("Barcode Entry", "Sample ID:")

        # Save Result CSV
        with open(result_csv_filename, mode='a', newline='') as result_file:
            fieldnames = ["Sample ID","Average_Length_mm", "Average_Width_mm", "Total_Seeds", "Total_Weight", "Thousand_Kernel_Weight", "Latitude", "Longitude", "GPS_UTC", "GPS_Source"]
            # fieldnames = ["Sample ID", "Average_Length_mm", "Average_Width_mm", "Total_Length_mm", "Total_Width_mm", "Total_Seeds", "Total_Weight", "Thousand_Kernel_Weight"]
            writer = csv.DictWriter(result_file, fieldnames=fieldnames)
            gps_lat = ""
            gps_lon = ""
            gps_utc = ""
            gps_source = ""

            try:
                gps = GPSReader()  # defaults to COM5
                fix = gps.get_fix(max_wait_sec=30)
                gps.close()

                if fix:
                    gps_lat = fix.get("latitude", "")
                    gps_lon = fix.get("longitude", "")
                    gps_utc = fix.get("utc", "")
                    gps_source = fix.get("source", "")
            except Exception as e:
                logging.error(f"GPS error: {e}", exc_info=True)
            writer.writerow({
                "Sample ID":self.sample_ID,
                "Average_Length_mm": self.avg_length,
                "Average_Width_mm": self.avg_width,
                # "Total_Length_mm": self.total_length_mm,
                # "Total_Width_mm": self.total_width_mm,
                "Total_Seeds": self.seed_count,
                "Total_Weight": self.seed_weight,
                "Thousand_Kernel_Weight": round(self.TKW, 3),
                "Latitude": gps_lat,
                "Longitude": gps_lon,
                "GPS_UTC": gps_utc,
                "GPS_Source": gps_source,
            })
        print(f"[INFO] Results saved to {result_csv_filename}")

        #Adding images to the results
        # original_image_filename = os.path.join(run_folder, f"original_image_sample_{self.sample_ID}.jpg")
        masked_image_filename = os.path.join(run_folder, f"masked_image_sample_{self.sample_ID}.jpg")
        # cv2.imwrite(original_image_filename, cv2.resize(self.image, self.image_size))
        cv2.imwrite(masked_image_filename, self.out_with_seed_id)

        self.save_results_btn.pack_forget()
        self.saving_label.config(text="Results saved !!!!",font=("Arial", 10,"bold"))


if __name__ == "__main__":
    app = CameraApp()
    try:
        app.mainloop()
    except  Exception:
        traceback.print_exc()

