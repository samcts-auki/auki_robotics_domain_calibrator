import numpy as np
import cv2
import sys
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from .domain import Domain
from .marker_calibrator import *

T_domain_to_ROS = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

T_conversion = T_domain_to_ROS

last_domain = None


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    :param config_path: Path to config file. If None, looks for config.yaml in parent directory (python/).
    :return: Dictionary with configuration
    """
    if config_path is None:
        # Look for config.yaml in the parent directory (python/)
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def rpy_from_matrix_scipy(mat, degrees=False):
    M = np.array(mat, dtype=float)
    if M.shape == (4,4):
        M = M[:3,:3]
    rot = R.from_matrix(M)
    # SciPy returns [z, y, x] for 'zyx', which correspond to yaw, pitch, roll.
    yaw, pitch, roll = rot.as_euler('zyx', degrees=degrees)
    # Reorder to match ROS (roll, pitch, yaw)
    return roll, pitch, yaw


class DomainCalibratorPy:
    def __init__(self, domain_config=None, camera_matrix=None, dist_coeffs=None, config_path=None):
        """
        Initialize the domain calibrator.
        
        :param domain_config: dict with app_key, app_secret, map_endpoint, etc.
                          If None, will load from config file
        :param camera_matrix: 3x3 numpy array camera intrinsic matrix
                             If None, will try to load from config
        :param dist_coeffs: distortion coefficients array
                           If None, will try to load from config
        :param config_path: Path to config YAML file (optional)
        """
        # Load config if domain_config not provided
        if domain_config is None:
            config = load_config(config_path)
            domain_config = {
                'app_key': config['auth']['app_key'],
                'app_secret': config['auth']['app_secret'],
                'api_base_url': config['auth'].get('api_base_url', 'https://api.auki.network'),
                'dds_base_url': config['auth'].get('dds_base_url', 'https://dds.auki.network'),
                'map_endpoint': config['domain'].get('map_endpoint', 'https://dsc.dev.aukiverse.com/spatial/crosssection')
            }
            
            # Load camera matrix and dist_coeffs from config if not provided
            if camera_matrix is None and config.get('camera', {}).get('camera_matrix'):
                camera_matrix = np.array(config['camera']['camera_matrix'], dtype=np.float64)
            
            if dist_coeffs is None and config.get('camera', {}).get('dist_coeffs'):
                dist_coeffs = np.array(config['camera']['dist_coeffs'], dtype=np.float64)
            
            # Load portal conversion matrix from config (default identity)
            portal_conversion_matrix = np.eye(4)
            if config.get('calibration', {}).get('portal_conversion_matrix'):
                portal_conversion_matrix = np.array(config['calibration']['portal_conversion_matrix'])
            # Load camera axis conversion (applied to camera pose only)
            camera_axis_conversion = None
            if config.get('calibration', {}).get('camera_axis_conversion'):
                camera_axis_conversion = np.array(config['calibration']['camera_axis_conversion'])
        else:
            portal_conversion_matrix = np.eye(4)
            camera_axis_conversion = None
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Setup Domain
        self.domain = Domain(domain_config, portal_conversion_matrix=portal_conversion_matrix)
        ret, msg = self.domain.auth()
        if not ret:
            raise Exception(f"domain authentication failed. message: {msg}")
        print(f"domain authenticated {ret} {msg}")

        # Calibrator
        self.marker_calibrator = Calibrator()

        # Cache domain lookups to avoid repeated network calls
        self._domain_id_cache = {}
        self._authenticated_domain_id = None

        # Fixed max side for detection (matches reference behavior)
        self._detect_max_side = 2056
        # Optional camera-axis conversion (applied to camera pose only)
        self._camera_axis_conversion = camera_axis_conversion

    def detect(self, cv_image):
        """
        Detect QR codes in image and return camera pose estimates.
        
        :param cv_image: OpenCV image (BGR format)
        :return: List of transform matrices (T_Map_Camera) or None
        """
        global last_domain
        
        if self.camera_matrix is None:
            raise ValueError("Camera matrix not set")
        
        # Preprocess Image
        image, scale = self.marker_calibrator.preprocess_image(cv_image, self._detect_max_side)

        # Scale camera intrinsics to match resized image
        scaled_camera_matrix = self.camera_matrix.copy()
        scaled_camera_matrix[0, 0] *= scale  # fx
        scaled_camera_matrix[1, 1] *= scale  # fy
        scaled_camera_matrix[0, 2] *= scale  # cx
        scaled_camera_matrix[1, 2] *= scale  # cy
        
        # Detect Markers
        detected_markers = []
        markers = self.marker_calibrator.detect_markers(image)
        
        for marker in markers:
            if marker['value'] is None:
                print(f"Warning: unable to decode qr, value: {marker['value']}")
                continue

            if "HTTPS://R8.HR/" not in marker['value']:
                print(f"Warning: detected qr is not a portal, decoding: {marker['value']}")
                continue

            short_id = marker['value'].replace("HTTPS://R8.HR/", "")
            print(f"Detected Portal: {short_id}")

            if short_id in self._domain_id_cache:
                domain_id = self._domain_id_cache[short_id]
            else:
                domain_id = self.domain.get_domain_id(short_id)
                self._domain_id_cache[short_id] = domain_id

            if domain_id is None:
                print(f"Warning: failed to get domain id for {short_id}")
                continue

            # Authenticate domain on first detection or when domain changes
            if self._authenticated_domain_id != domain_id:
                if last_domain is not None and last_domain != domain_id:
                    print(f"Warning: domain changed from {last_domain} to {domain_id}")
                last_domain = domain_id
                ret, msg = self.domain.auth_domain(domain_id)

                if not ret:
                    print(f"Warning: failed to authenticate domain {domain_id}: {msg}")
                    continue
                
                self.domain.fetch_portal_poses()
                self._authenticated_domain_id = domain_id

            portal = self.domain.portals().get(short_id, None)
            if portal is None:
                print(f"Warning: {short_id} not found in domain domain_id: {domain_id}")
                continue
            
            # zbar returns 4 corners = 8 values (x1, y1, x2, y2, x3, y3, x4, y4)
            if len(marker['landmarks']) == 8:
                corners = np.array(list(zip(marker['landmarks'][::2], marker['landmarks'][1::2])), dtype=np.float32)
                tf_matrix = self.marker_calibrator.portal_pose(
                    portal,
                    corners,
                    scaled_camera_matrix,
                    self.dist_coeffs
                )
                if tf_matrix is not None:
                    detected_markers.append({
                        "T_Camera_QR": tf_matrix,
                        "short_id": short_id,
                        "corners": corners
                    })
                else:
                    print(f"Warning: failed to calibrate for portal {short_id}")
            else:
                print(f"Warning: QR code corners not detected correctly, got {len(marker['landmarks'])} values")

        return detected_markers
    
    def detect_and_calibrate(self, cv_image):
        """
        Detect QR codes in image and return camera pose estimates.
        
        :param cv_image: OpenCV image (BGR format)
        :return: List of transform matrices (T_Map_Camera) or None
        """
        global last_domain
        
        if self.camera_matrix is None:
            raise ValueError("Camera matrix not set")
        
        # Preprocess Image
        height, width, _ = cv_image.shape
        image, scale = self.marker_calibrator.preprocess_image(cv_image, self._detect_max_side)

        # Scale camera intrinsics to match resized image
        scaled_camera_matrix = self.camera_matrix.copy()
        scaled_camera_matrix[0, 0] *= scale  # fx
        scaled_camera_matrix[1, 1] *= scale  # fy
        scaled_camera_matrix[0, 2] *= scale  # cx
        scaled_camera_matrix[1, 2] *= scale  # cy
        
        # Detect Markers
        tfs = []
        markers = self.marker_calibrator.detect_markers(image)
        
        for marker in markers:
            if marker['value'] is None:
                print(f"Warning: unable to decode qr, value: {marker['value']}")
                continue

            if "HTTPS://R8.HR/" not in marker['value']:
                print(f"Warning: detected qr is not a portal, decoding: {marker['value']}")
                continue

            short_id = marker['value'].replace("HTTPS://R8.HR/", "")
            print(f"Detected Portal: {short_id}")

            if short_id in self._domain_id_cache:
                domain_id = self._domain_id_cache[short_id]
            else:
                domain_id = self.domain.get_domain_id(short_id)
                self._domain_id_cache[short_id] = domain_id

            if domain_id is None:
                print(f"Warning: failed to get domain id for {short_id}")
                continue

            # Authenticate domain on first detection or when domain changes
            if self._authenticated_domain_id != domain_id:
                if last_domain is not None and last_domain != domain_id:
                    print(f"Warning: domain changed from {last_domain} to {domain_id}")
                last_domain = domain_id
                ret, msg = self.domain.auth_domain(domain_id)

                if not ret:
                    print(f"Warning: failed to authenticate domain {domain_id}: {msg}")
                    continue
                
                self.domain.fetch_portal_poses()
                self._authenticated_domain_id = domain_id

            portal = self.domain.portals().get(short_id, None)
            if portal is None:
                print(f"Warning: {short_id} not found in domain domain_id: {domain_id}")
                continue
            
            # zbar returns 4 corners = 8 values (x1, y1, x2, y2, x3, y3, x4, y4)
            if len(marker['landmarks']) == 8:
                corners = np.array(list(zip(marker['landmarks'][::2], marker['landmarks'][1::2])), dtype=np.float32)
                tf_matrix = self.marker_calibrator.camera_pose(
                    portal,
                    corners,
                    scaled_camera_matrix,
                    self.dist_coeffs
                )
                if tf_matrix is not None:
                    tfs.append(tf_matrix)
                else:
                    print(f"Warning: failed to calibrate for portal {short_id}")
            else:
                print(f"Warning: QR code corners not detected correctly, got {len(marker['landmarks'])} values")

        if len(tfs) < 1:
            print("No Valid Markers Detected")
            return None

        # Average the transforms
        T_Map_Camera_avg = average_transforms(tfs)
        # Axis conversion is now applied in marker_calibrator.py, so no need to apply here
        return T_Map_Camera_avg

    def get_map(self, resolution=20, domain_id=None, image_format='png'):
        """
        Get map for the domain.
        
        :param resolution: Pixels per meter (default: 20)
        :param domain_id: Domain ID (default: None, uses current domain)
        :param image_format: Image format - 'png', 'bmp', 'pgm' (default: 'png')
        Note: For 'pgm', requests PNG from server and converts to PGM
        """
        if domain_id is None:
            domain_id = last_domain
        return self.domain.get_map(resolution=resolution, domain_id=domain_id, image_format=image_format)
