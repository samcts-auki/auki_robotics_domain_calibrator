import cv2
import numpy as np
from pyzbar import pyzbar
import qr_lab
import auki_pnplab
import logging
from scipy.spatial.transform import Rotation as R
import os
from contextlib import redirect_stderr

logger = logging.getLogger(__name__)

# Default detection/pose method for the whole module. "auki" uses qr-lab
# (detection) + pnp-lab (PnP), both pure-Rust with no zbar/OpenCV-solvePnP
# runtime dependency. "pyzbar" is the legacy zbar + cv2.solvePnP path, kept
# for comparison and as a fallback.
DEFAULT_METHOD = "auki"

# pnp-lab's estimate_square_pose_from_pixels() returns the marker pose
# expressed in an OpenGL-convention camera frame (Y-up, Z-backward/toward
# viewer). This project's downstream conventions (Domain portal poses, the
# `T_domain_to_ROS` conversion in domain_calibrator.py) are built around
# cv2.solvePnP's OpenCV camera frame (Y-down, Z-forward). Flipping Y and Z
# is a pure change of camera-frame basis (both frames share the same origin,
# the camera's optical center) so it is applied as a LEFT multiply.
_GL_TO_CV_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])

# Same object-local axis correction the legacy cv2/pyzbar path applies after
# solvePnP (see `_portal_pose_cv2`): swap X/Y, negate Z. Applying it
# identically after the pnp-lab solve keeps both methods' output numerically
# compatible, since both share the same TL/TR/BR/BL square object model.
_QR_LOCAL_AXIS_FIX = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


class Calibrator():
    def __init__(self) -> None:
        pass

    def detect_markers(self, img, method=DEFAULT_METHOD):
        """
        Detect QR codes in the image.
        Returns a list of markers with their decoded values and corner points.

        :param method: "auki" (default, qr-lab) or "pyzbar" (legacy zbar).
        """
        if method == "auki":
            return self._detect_markers_auki(img)
        if method == "pyzbar":
            return self._detect_markers_pyzbar(img)
        raise ValueError(f"unknown detection method: {method!r} (expected 'auki' or 'pyzbar')")

    def _detect_markers_auki(self, img):
        """
        Detect QR codes using qr-lab (pure Rust, no zbar dependency).
        Returns markers in the same shape as `_detect_markers_pyzbar`.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray = np.ascontiguousarray(gray, dtype=np.uint8)

        result = qr_lab.scan(gray, preset="robust_fast", refine=True)

        markers = []
        for detection in result.get("codes", []):
            # Prefer subpixel-refined corners (present since refine=True was
            # requested) over the coarse module-region corners -- better
            # input for the PnP solve.
            corners = detection.get("refined_corners_source") or detection.get("corners_source")
            if corners is None or len(corners) != 4:
                continue

            # qr-lab already returns corners in TL, TR, BR, BL source-pixel
            # order (same convention pyzbar's orientation-corrected points
            # use below), so no reordering is needed here.
            landmarks = []
            for x, y in corners:
                landmarks.extend([float(x), float(y)])

            markers.append(
                {
                    'type': 0,  # QR code (matching original type)
                    'value': detection['code']['payload'],
                    'landmarks': landmarks
                }
            )

        return markers

    def _detect_markers_pyzbar(self, img):
        """
        Detect QR codes in the image using zbar (via pyzbar). Legacy path.
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

    def portal_pose(self, portal, corners, camera_matrix, dist_coeffs, method=DEFAULT_METHOD):
        """
        Estimate T_Camera_QR (4x4) for a single portal/marker detection.

        :param method: "auki" (default, pnp-lab) or "pyzbar" (legacy cv2.solvePnP).
        """
        if method == "auki":
            return self._portal_pose_auki(portal, corners, camera_matrix, dist_coeffs)
        if method == "pyzbar":
            return self._portal_pose_cv2(portal, corners, camera_matrix, dist_coeffs)
        raise ValueError(f"unknown pose method: {method!r} (expected 'auki' or 'pyzbar')")

    def _portal_pose_cv2(self, portal, corners, camera_matrix, dist_coeffs):
        """Legacy pose solve via cv2.solvePnP(SOLVEPNP_IPPE_SQUARE)."""
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
        T_Camera_QR = T_Camera_QR @ _QR_LOCAL_AXIS_FIX
        # print(f"T_Camera_QR:\n{T_Camera_QR}")
        return T_Camera_QR

    def _portal_pose_auki(self, portal, corners, camera_matrix, dist_coeffs):
        """Pose solve via pnp-lab's estimate_square_pose_from_pixels (no OpenCV)."""
        camera = {
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
            "dist": [float(d) for d in np.asarray(dist_coeffs).flatten()] if dist_coeffs is not None else [],
        }
        pixels = np.asarray(corners, dtype=np.float64)

        try:
            estimate = auki_pnplab.estimate_square_pose_from_pixels(pixels, portal['size'], camera)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"pnp-lab square pose estimation failed: {e}")
            return None

        pose = estimate["pose"]
        position = pose["position"]
        rotation = pose["rotation"]
        rot_matrix = R.from_quat([rotation["x"], rotation["y"], rotation["z"], rotation["w"]]).as_matrix()

        T_gl_object = np.eye(4, dtype=np.float64)
        T_gl_object[:3, :3] = rot_matrix
        T_gl_object[:3, 3] = [position["x"], position["y"], position["z"]]

        # pnp-lab's pose is expressed in an OpenGL camera frame; convert to
        # OpenCV camera frame, then apply the same object-local axis fix the
        # legacy cv2 path applies, so both methods share one output contract.
        T_Camera_QR = _GL_TO_CV_CAMERA @ T_gl_object
        # print(f"T_Camera_QR:\n{T_Camera_QR}")
        return T_Camera_QR

    def camera_pose(self, portal, corners, camera_matrix, dist_coeffs, method=DEFAULT_METHOD):
        T_Camera_QR = self.portal_pose(portal, corners, camera_matrix, dist_coeffs, method=method)
        if T_Camera_QR is None:
            return None

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
