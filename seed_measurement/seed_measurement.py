import cv2
import numpy as np

class seed_measurement:

    def __init__(self,img_path, aruco_size_mm=20.0):
        self.img_path = img_path
        self.aruco_size = aruco_size_mm
        pass

    def __calculate_pixel_to_mm_scale(self):
        gray = cv2.cvtColor(self.img_path, cv2.COLOR_BGR2GRAY)

        # Setup ArUco detector
        aruco = cv2.aruco
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)  # choose the dict you used to print marker
        parameters = aruco.DetectorParameters_create()

        corners_list, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if corners_list is None or len(corners_list) == 0:
            raise RuntimeError("No ArUco markers detected")

        # If multiple markers, pick the one you want. Here we use the first detected marker:
        corners = corners_list[0].reshape((4, 2))  # shape (4,2): four corners in pixel coords

        # Optional: refine corners with cornerSubPix for subpixel accuracy
        # reshape for cornerSubPix: (N,1,2)
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, corners.reshape(-1,1,2), (5,5), (-1,-1), term)

        # Compute lengths of the 4 sides (pixel units)
        side_lengths = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            side_lengths.append(np.linalg.norm(p1 - p2))
        mean_side_px = float(np.mean(side_lengths))

        # pixel to mm
        px_to_mm = self.aruco_size_mm / mean_side_px  # mm per pixel

        return {
            "corners_px": corners,      # (4,2)
            "mean_side_px": mean_side_px,
            "px_to_mm": px_to_mm,
            "ids": ids
        }
    
    def calculate_length_width_in_mm():

        pxl_mm_scale = __calculate_pixel_to_mm_scale()
        pass
