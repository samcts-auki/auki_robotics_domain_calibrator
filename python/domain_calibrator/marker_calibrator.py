import cv2
import numpy as np
from pyzbar import pyzbar
import logging
from scipy.spatial.transform import Rotation as R
import os
from contextlib import redirect_stderr

class Calibrator():
    def __init__(self) -> None:
        pass

    def detect_markers(self, img):
        """
        Detect QR codes in the image using zbar (via pyzbar).
        Returns a list of markers with their decoded values and corner points.
        """
        # pyzbar expects grayscale images
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Decode QR codes
        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull):
                decoded_objects = pyzbar.decode(gray)
        
        markers = []
        for obj in decoded_objects:
            # print(obj)
            if obj.type != "QRCODE":
                continue

            # Get the decoded data
            value = obj.data.decode('utf-8')
            
            # Get the polygon points (corners of the QR code)
            # pyzbar returns points as a list of Point objects
            points = obj.polygon
            
            # Extract corner points - zbar typically returns 4 corners
            # We need to format them as [x1, y1, x2, y2, x3, y3, x4, y4]
            landmarks = []
            if len(points) != 4:
                # invalid number of points
                continue

            orient = obj.orientation

            if orient == 'DOWN':
                points = [points[0], points[1], points[2], points[3]]
            elif orient == 'LEFT':
                points = [points[3], points[0], points[1], points[2]]
            elif orient == 'UP':
                points = [points[2], points[3], points[0], points[1]]
            elif orient == 'RIGHT':
                points = [points[1], points[2], points[3], points[0]]
            else:
                # Fallback if orientation is 'UNKNOWN'
                points = points

            # Use the first 4 points as corners
            for i in range(4):
                landmarks.extend([points[i].x, points[i].y])

            markers.append(
                {
                    'type': 0,  # QR code (matching original type)
                    'value': value,
                    'landmarks': landmarks
                }
            )
        
        return markers

    def preprocess_image(self, img, target_size=2056):
        # Handle both grayscale (2D) and color (3D) images
        if len(img.shape) == 2:
            height, width = img.shape
            channels = 1
        else:
            height, width, channels = img.shape
            if channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Determine the scale factor
        max_side = max(width, height)

        # Only downscale to the fixed max side (avoid upscaling)
        scale_factor = min(1.0, target_size / max_side)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return img, scale_factor

    def portal_pose(self, portal, corners, camera_matrix, dist_coeffs):
        # self.logger.debug(f"portal_size: {portal['size']}")
        half_size = portal['size'] / 2
        # 3D points of the QR code in its own coordinate system (Z=0 plane)
        qr_world_coords = np.array([
            [-half_size,  half_size, 0],  # Top-left
            [ half_size,  half_size, 0],  # Top-right
            [ half_size, -half_size, 0],  # Bottom-right
            [-half_size, -half_size, 0]   # Bottom-left
        ], dtype=np.float32)
        # Estimate pose
        success, rvec, tvec = cv2.solvePnP(qr_world_coords, corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

        if not success:
            return None
        
        R, _ = cv2.Rodrigues(rvec)
        T_Camera_QR = np.eye(4, dtype=np.float64)  # Initialize as identity
        T_Camera_QR[:3, :3] = R
        T_Camera_QR[:3, 3] = tvec.flatten()
        T_Camera_QR = T_Camera_QR @ np.array([
                                        [-1.0,0.0,0.0,0.0],
                                        [0.0,1.0,0.0,0.0],
                                        [0.0,0.0,-1.0,0.0],
                                        [0.0,0.0,0.0,1.0]
                                    ])
        return T_Camera_QR
    
    def camera_pose(self, portal, corners, camera_matrix, dist_coeffs):
        T_Camera_QR = self.portal_pose(portal, corners, camera_matrix, dist_coeffs)

        # T Domain Camera = T_Domain_QR * T_QR_Camera
        T_Domain_Camera = portal['pose'] @ np.linalg.inv(T_Camera_QR)

        return T_Domain_Camera # T_Reference_Object
    

def average_transforms(transforms):
    rotations = []
    translations = []

    for T in transforms:
        R_mat = T[:3, :3]
        t_vec = T[:3, 3]
        rotations.append(R_mat)
        translations.append(t_vec)

    # Convert to rotation objects
    rot_objs = R.from_matrix(rotations)

    # Average the rotations using quaternion averaging
    mean_rot = rot_objs.mean().as_matrix()

    # Average the translations
    mean_trans = np.mean(translations, axis=0)

    # Construct average transform
    T_avg = np.eye(4)
    T_avg[:3, :3] = mean_rot
    T_avg[:3, 3] = mean_trans
    return T_avg
