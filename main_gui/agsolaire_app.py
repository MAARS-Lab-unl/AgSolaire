import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO, SAM
import numpy as np
import cv2
import os
import sys

# ======== adding path for other modules ========
module_path = "/home/herve/agsolaire_ml_UNL"
sys.path.insert(0,module_path)

from read_weight import scale_reader
from seed_measurement import seed_measurement

class CameraApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Seed Segmentation Capture")
        self.geometry("1000x700")
        self.resizable(False, False)

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

        camera_path = "/dev/video2"

        self.controller = controller
        self.cap = cv2.VideoCapture(camera_path)
        self.label = tk.Label(self)
        self.label.pack()

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        capture_btn = ttk.Button(btn_frame, text="Capture Image", command=self.capture_image)
        capture_btn.grid(row=0, column=0, padx=10)

        quit_btn = ttk.Button(btn_frame, text="Quit", command=self.quit_app)
        quit_btn.grid(row=0, column=1, padx=10)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.current_frame = frame
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
            filename = "captured_image.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.controller.captured_image_path = filename
            self.controller.show_frame(ResultScreen)

    def quit_app(self):
        self.cap.release()
        self.controller.destroy()


class ResultScreen(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.image_label = tk.Label(self)
        self.image_label.pack(pady=20)

        btn_frame = tk.Frame(self)
        btn_frame.pack()

        back_btn = ttk.Button(btn_frame, text="Back to Camera", command=self.go_back)
        back_btn.grid(row=0, column=0, padx=10)

        infer_btn = ttk.Button(btn_frame, text="Send for Inference", command=self.run_inference)
        infer_btn.grid(row=0, column=1, padx=10)

        self.status_label = tk.Label(self, text="", font=("Arial", 12))
        self.status_label.pack(pady=10)

        # ======== configuring path for the DL models ========
        self.yolo_model = YOLO("/home/herve/Downloads/best.pt")
        self.sam_model = SAM("sam2_b.pt")


    def tkraise(self, *args, **kwargs):
        """Overridden to refresh image each time screen is shown"""
        super().tkraise(*args, **kwargs)
        self.show_captured_image()

    def show_captured_image(self):
        path = self.controller.captured_image_path
        if path and os.path.exists(path):
            img = Image.open(path).resize((800, 600))
            imgtk = ImageTk.PhotoImage(img)
            self.image_label.imgtk = imgtk
            self.image_label.config(image=imgtk)
        else:
            self.image_label.config(text="No image captured yet.")

    def go_back(self):
        self.controller.show_frame(CameraScreen)

    def run_inference(self):
        # Placeholder: integrate your YOLO + SAM2 inference here
        

        image = self.controller.captured_image_path
        image = cv2.imread(image)
        
        image = cv2.resize(image, (640,480))

        # ======== weight measurement ========
        weight_reading = scale_reader.scale_reader()
        seed_weight = weight_reading.read_weight()

        # ======== mm per pixel scale calculation ========
        pixel_to_conversion = seed_measurement.seed_measurement(self.controller.captured_image_path)

        scale_calculation = pixel_to_conversion.calculate_length_width_in_mm()

        if scale_calculation is not None:

            PX_PER_MM = scale_calculation['mm_per_pixel']
        else:
            
            PX_PER_MM = None


        # Run YOLO detection
        results = self.yolo_model.predict(source=image, conf=0.4)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        print(f"Number of seeds: {len(boxes)}")

        # Run SAM segmentation using YOLO boxes as prompts
        sam_results = self.sam_model(image, bboxes=boxes)

        # self.after(2000, lambda: self.status_label.config(text="Inference complete (placeholder)."))
        overlay = image.copy()
        seed_id = 0

        for r in sam_results:
            masks = r.masks.data.cpu().numpy()  # [N, H, W]

            for mask in masks:
                seed_id += 1
                mask = (mask > 0.5).astype(np.uint8)  # ensure binary mask

                # Find contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                cnt = max(contours, key=cv2.contourArea)

                # # ---- Fit ellipse instead of rectangle ----
                # if len(cnt) < 5:   # cv2.fitEllipse requires >=5 points
                #     continue

                ellipse = cv2.fitEllipse(cnt)   # (center(x,y), (major_axis, minor_axis), angle)
                (cx, cy), (MA, ma), angle = ellipse

                # Major = length, Minor = width
                length_px = max(MA, ma)
                width_px  = min(MA, ma)

                # Convert to mm if calibration known
                length_mm = length_px / PX_PER_MM if PX_PER_MM else None
                width_mm  = width_px  / PX_PER_MM if PX_PER_MM else None

                # ---- Draw ellipse ----
                cv2.ellipse(overlay, ellipse, (0, 255, 0), 2)  # blue ellipse
                cv2.circle(overlay, (int(cx), int(cy)), 4, (0, 255, 0), -1)  # green center

                # Put text label
                if length_mm and width_mm:
                    label = f"{seed_id}: {length_mm:.2f} x {width_mm:.2f} mm"
                else:
                    label = f"{seed_id}: {length_px:.1f} x {width_px:.1f} px"

                # cv2.putText(overlay, label, (int(cx)+10, int(cy)-10),
                            # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Blend overlay with original image
        out = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        inference_image = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
        inference_image = Image.fromarray(inference_image)

        # img = Image.open(out).resize((800, 600))
        imgtk = ImageTk.PhotoImage(inference_image)
        self.image_label.imgtk = imgtk
        self.image_label.config(image=imgtk)

        seed_count_string = f"seed count: {len(boxes[0])}"
        weight_string = f"Weight: {seed_weight}"
        self.status_label.config(text=seed_count_string )
        self.status_label.config(text=weight_string)


if __name__ == "__main__":
    app = CameraApp()
    app.mainloop()


