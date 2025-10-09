import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import os

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
        self.status_label.config(text="Running inference...")
        self.after(2000, lambda: self.status_label.config(text="Inference complete (placeholder)."))


if __name__ == "__main__":
    app = CameraApp()
    app.mainloop()


