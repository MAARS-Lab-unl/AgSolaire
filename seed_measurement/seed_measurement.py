import cv2
import numpy as np

class seed_measurement:

    def __init__(self,img_path, aruco_size_mm=20.0):
        self.img_path = img_path
        self.aruco_size_mm = aruco_size_mm
        pass

    def __calculate_pixel_to_mm_scale(self):
        gray = cv2.cvtColor(self.img_path, cv2.COLOR_BGR2GRAY)

        # Setup ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  # choose the dict you used to print marker
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict,parameters)

        corners_list, ids, rejected = detector.detectMarkers(gray)

        #checking if no marker is present
        if corners_list is None or len(corners_list) == 0:
            return None
            # raise RuntimeError("No ArUco markers detected")

        #debugging aruco detection
        # if ids is not None:
        #     cv2.aruco.drawDetectedMarkers(self.img_path,corners_list, ids)
        #     cv2.imshow("markers",self.img_path)

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

        # mm per pixel calculation
        mm_per_pxl = self.aruco_size_mm / mean_side_px  # mm per pixel

        #return a dictionary
        return {
            "corners_px": corners,      
            "mean_side_px": mean_side_px,
            "mm_per_pixel": mm_per_pxl,
            "ids": ids
        }
    
    def calculate_length_width_in_mm(self):
        mm_per_pxl_scale = self.__calculate_pixel_to_mm_scale()
        return mm_per_pxl_scale
        
if __name__ == '__main__':
    cap = cv2.VideoCapture("/dev/video2")
    while True:
        ret,frame = cap.read()
        img = frame
        # img = cv2.imread("image_with_marker.jpg")
        cv2.imshow("testing",img)
        obj = seed_measurement(img)
        res = obj.calculate_length_width_in_mm()
        # print(res)
        if res is not None:
            print(f"Mean marker side (px): {res['mean_side_px']:.2f}")
            print(f"Scale: {res['px_to_mm']:.6f} mm / px")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()