import tkinter as tk
from tkinter import ttk,font
from ctypes import windll
from PIL import Image, ImageTk
from ultralytics import YOLO, SAM, FastSAM
import numpy as np
import torch
import cv2
import csv
import os
import sys
import usb.core
import threading
from datetime import datetime

import traceback, logging

# ======== adding path for other modules ========
    
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

        camera_index = 0
        if self.find_camera_index() != None:
            camera_index = self.find_camera_index()[0]

        self.controller = controller
        self.cap = cv2.VideoCapture(camera_index)
        self.label = tk.Label(self)
        self.label.pack(pady = 20)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        ttk.Style().configure('My.TButton', 
        font=('Helvetica', 15),  # Increase font size
        padding=(10, 5)) 

        capture_btn = ttk.Button(btn_frame, text="Capture Image", command=self.capture_image,style='My.TButton')
        capture_btn.grid(row=0, column=0, padx=5)

        quit_btn = ttk.Button(btn_frame, text="Quit", command=self.quit_app, style='My.TButton')
        quit_btn.grid(row=0, column=1, padx=5)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # frame = cv2.flip(frame, 1)
            self.current_frame = frame
            frame = cv2.resize(frame, (800,600))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
            #debugging
            # print(f"width: {imgtk.width()} \n Height {imgtk.height()}")
        self.after(10, self.update_frame)

    def capture_image(self):
        if hasattr(self, 'current_frame'):
            filename = "captured_image.png"
            cv2.imwrite(filename, self.current_frame)
            self.controller.captured_image_path = filename
            self.controller.show_frame(ResultScreen)

    def quit_app(self):
        self.cap.release()
        self.controller.destroy()
    
    def find_camera_index(self):
        """Check if a USB device with given VID/PID is connected."""
        #camera vendor ID and Product ID
        VID = 0X0C45
        PID = 0x6366
        dev = usb.core.find(idVendor=VID, idProduct=PID) 
        camera_indices = list()

        if dev:
            print("A Camera is found!")
            
            for i in range(10):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # use DirectShow backend
                if cap.isOpened():
                    ret,frame = cap.read()
                    if ret:
                        cap.release()
                        camera_indices.append(i)
                    else:
                        continue
                cap.release()
            return camera_indices
                      
        else:
            print("NO CAMERA FOUND!!, PLEASE CHECK IF THE CAMERA IS CONNECTED")
            return None


class ResultScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        image_frame = tk.Frame(self)
        image_frame.pack(side="left", anchor="nw", pady=20, padx=20)

        self.controls_frame = tk.Frame(self)
        self.controls_frame.pack(side="left",  anchor="nw")

        self.image_label = tk.Label(image_frame)
        self.image_label.pack()

        btn_frame = tk.Frame(self.controls_frame)
        btn_frame.pack()


        ttk.Style().configure('My.TButton', 
        font=('Helvetica', 10),  # Increase font size
        padding=(10, 5))  
        
        back_btn = ttk.Button(btn_frame, text="Back to Camera", command=self.go_back, style='My.TButton')
        back_btn.pack( pady = 20, padx=20, side ="left")

        infer_btn = ttk.Button(btn_frame, text="Send for Inference", command=self.run_inference_button_click, style='My.TButton')
        infer_btn.pack(pady = 20, padx=10)

        results_label = tk.Frame(self.controls_frame)
        results_label.pack()

        self.spinner = Spinner(self.controls_frame, size=60, color="#4CAF50", speed=20)

        self.inference_label = tk.Label(results_label, text="", font=("Arial", 12))

        self.count_label = tk.Label(results_label, text="", font=("Arial", 12))
        self.count_label.pack(pady=10)

        self.weight_label = tk.Label(results_label, text="", font=("Arial", 12))
        self.weight_label.pack(pady=10)

        self.TKW_label = tk.Label(results_label, text="", font=("Arial", 12))
        self.TKW_label.pack(pady=10)
        
        self.mm_to_pxl_label = tk.Label(results_label, text="", font=("Arial", 12))
        self.mm_to_pxl_label.pack(pady=10)

        
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

        self.inference_label.config(text="")

        self.spinner.pack_forget()

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

        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
            self.yolo_model = YOLO(os.path.join(base_path, "model", "yolo_detection_model.pt"))
            self.sam_model = SAM(os.path.join(base_path, "model", "mobile_sam.pt"))
        else:
            base_path = os.path.dirname(os.path.dirname(__file__))
            self.yolo_model = YOLO(os.path.join(base_path, "model", "yolo_detection_model.pt"))
            self.sam_model = SAM(os.path.join(base_path, "model", "mobile_sam.pt"))


        image = self.controller.captured_image_path
        image = cv2.imread(image)
        image = cv2.resize(image, (1024, 1024))

        # Weight Measurement
        weight_reading = scale_reader.scale_reader()
        seed_weight_wt_unit = weight_reading.read_weight()
        seed_weight = weight_reading.read_weight_as_value()

        # mm per pixel scale calculation
        pixel_to_conversion = seed_measurement(cv2.imread(self.controller.captured_image_path))
        scale_calculation = pixel_to_conversion.calculate_length_width_in_mm()
        PX_PER_MM = scale_calculation['mm_per_pixel'] if scale_calculation else None

        # YOLO Detection
        results = self.yolo_model.predict(source=image, conf=0.6)
        boxes = results[0].boxes.xywh.cpu().numpy()
        seed_count = len(boxes)

        # Thousand Kernel Weight
        TKW = (seed_weight / seed_count) * 1000 if seed_count > 0 else 0

        # SAM Segmentation
        sam_results = self.sam_model.predict(image, points=boxes[:, :2])

        overlay = image.copy()
        seed_id = 0
        seed_data = []

        for r in sam_results:
            masks = r.masks.data.cpu().numpy()
            for mask in masks:
                seed_id += 1
                mask = (mask > 0.5).astype(np.uint8)  # ensure binary mask

                # Find contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                cnt = max(contours, key=cv2.contourArea)

                # ---- Fit ellipse instead of rectangle ----
                if len(cnt) < 5:   # cv2.fitEllipse requires >=5 points
                    continue

                ellipse = cv2.fitEllipse(cnt)   # (center(x,y), (major_axis, minor_axis), angle)
                (cx, cy), (MA, ma), angle = ellipse

                # Major = length, Minor = width
                length_px = max(MA, ma)
                width_px  = min(MA, ma)

                # Convert to mm if calibration known
                length_mm = length_px / PX_PER_MM if PX_PER_MM else None
                width_mm  = width_px  / PX_PER_MM if PX_PER_MM else None

                cv2.ellipse(overlay, ellipse, (0, 255,0), 4)
                cv2.circle(overlay, (int(cx), int(cy)), 4, (0, 255, 0), -1)

                seed_data.append({
                    "Seed_ID": seed_id,
                    "Length_mm": round(length_mm, 3) if length_mm else None,
                    "Width_mm": round(width_mm, 3) if width_mm else None
                })

        # ======== Create Timestamped Run Folder ========
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join(os.getcwd(), f"Results/Run_{run_timestamp}")
        os.makedirs(run_folder, exist_ok=True)
        print(f"[INFO] Run folder created: {run_folder}")

        # Save Seed Measurements CSV
        seed_csv_filename = os.path.join(run_folder, f"seed_measurements_{run_timestamp}.csv")
        with open(seed_csv_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["Seed_ID", "Length_mm", "Width_mm"])
            writer.writeheader()
            writer.writerows(seed_data)
        print(f"[INFO] Seed metrics saved to {seed_csv_filename}")

        # Compute Summary Statistics
        valid_lengths = [s["Length_mm"] for s in seed_data if s["Length_mm"]]
        valid_widths = [s["Width_mm"] for s in seed_data if s["Width_mm"]]

        avg_length = round(sum(valid_lengths) / len(valid_lengths), 3) if valid_lengths else 0
        avg_width = round(sum(valid_widths) / len(valid_widths), 3) if valid_widths else 0

        # Save Result CSV
        result_csv_filename = os.path.join(run_folder, f"result_{run_timestamp}.csv")
        with open(result_csv_filename, mode='w', newline='') as result_file:
            fieldnames = ["Average_Length_mm", "Average_Width_mm", "Total_Seeds", "Total_Weight", "Thousand_Kernel_Weight"]
            writer = csv.DictWriter(result_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                "Average_Length_mm": avg_length,
                "Average_Width_mm": avg_width,
                "Total_Seeds": seed_count,
                "Total_Weight": seed_weight_wt_unit,
                "Thousand_Kernel_Weight": round(TKW, 3)
            })
        print(f"[INFO] Results saved to {result_csv_filename}")

        # Display Overlay on GUI
        out = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        out = cv2.resize(out, self.image_size)
        inference_image = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
        inference_image = Image.fromarray(inference_image)

        # remove progressbar 
        self.spinner.pack_forget()
        
        #Update label
        self.inference_label.config(text="Inference Completed !!!")

        # img = Image.open(out).resize((800, 600))
        imgtk = ImageTk.PhotoImage(inference_image)
        self.image_label.imgtk = imgtk
        self.image_label.config(image=imgtk)

        seed_count_string = f"seed count: {seed_count}"
        weight_string = f"Weight: {seed_weight_wt_unit}"
        TKW_string = f"Thousand Kernel Weight: {TKW}"
        mm_to_pxl_string = f"mm/pxl: {PX_PER_MM}"

        self.count_label.config(text=seed_count_string )
        self.weight_label.config(text=weight_string)
        self.TKW_label.config(text=TKW_string)
        self.mm_to_pxl_label.config(text=mm_to_pxl_string)

        # Cleanup
        del sam_results
        del results
        del self.sam_model
        del self.yolo_model

        print("[INFO] Inference completed successfully.")


if __name__ == "__main__":
    app = CameraApp()
    try:
        app.mainloop()
    except  Exception:
        traceback.print_exc()

