import json
import re
import base64
from PIL import Image
import io
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading 
from datetime import datetime
import httpx
import logging
from typing import Optional
import math

from .http_utils import send_request, send_files

logger = logging.getLogger(__name__)

# Token expiration time (1 hour in seconds)
TOKEN_EXPIRATION_TIME = 3600


def transformation_matrix(trans, quat):
    # Convert quaternion to 3x3 rotation matrix
    rotation_matrix = R.from_quat(quat).as_matrix()
    # Construct 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = trans
    return T


class Domain:
    def __init__(self, domain_config, portal_conversion_matrix=None):
        """
        Initialize the Domain object.
        
        :param domain_config: dict type that contains domain configurations.
            Required keys: app_key, app_secret
            Optional keys: api_base_url, dds_base_url, map_endpoint
        :param portal_conversion_matrix: 4x4 numpy array for coordinate conversion
        """
        self.app_key = domain_config.get("app_key", "")
        self.app_secret = domain_config.get("app_secret", "")
        self.api_base_url = domain_config.get("api_base_url", "https://api.auki.network")
        self.dds_base_url = domain_config.get("dds_base_url", "https://dds.auki.network")
        self.map_endpoint = domain_config.get("map_endpoint", "https://dsc.dev.aukiverse.com/spatial/crosssection")
        self.path_endpoint = domain_config.get("path_endpoint", "https://dsc.auki.network/spatial/pathfind")
        self.restricted_dest_endpoint = domain_config.get("restricted_dest_endpoint", "https://dsc.auki.network/spatial/restricttonavmesh")
        self.robot_radius = domain_config.get("robot_radius", 0.2)
        self.override_domain_id = domain_config.get("override_domain_id", None)

        if not self.app_key or not self.app_secret:
            raise ValueError("app_key and app_secret are required in domain_config")
        
        self.client = httpx.Client(timeout=30.0)
        self._dds_token: Optional[str] = None
        self._domain_info: Optional[dict] = None
        self._domain_server: Optional[str] = None
        self._dds_token_timestamp: float = 0.0
        self._domain_info_timestamp: float = 0.0
        self._device_id = self._get_device_id()

        self._portals = {}
        self._portal_conversion_matrix = portal_conversion_matrix

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_every_hour, daemon=True)
        self._thread.start()

    def _get_device_id(self) -> str:
        """Generate or retrieve a unique device ID."""
        import uuid
        return f"domain-calibrator-{uuid.uuid4().hex[:8]}"

    def _run_every_hour(self):
        while not self._stop_event.is_set():
            # Refresh tokens periodically
            import time
            self._refresh_tokens_if_needed()
            # Wait one hour (3600 seconds) or until the stop event is set
            self._stop_event.wait(timeout=3600)

    def _refresh_tokens_if_needed(self):
        """Refresh tokens if they're expired or about to expire."""
        import time
        current_time = time.time()
        
        # Refresh DDS token if expired
        if not self._dds_token or (current_time - self._dds_token_timestamp) >= TOKEN_EXPIRATION_TIME:
            try:
                self._dds_token = self._get_dds_token()
                self._dds_token_timestamp = current_time
            except Exception as e:
                logger.error(f"Failed to refresh DDS token: {e}")

    def _get_dds_token(self) -> str:
        """Get App Domain Service Access Token using Basic Auth."""
        url = f"{self.api_base_url}/service/domains-access-token"
        
        # Encode app_key:app_secret to base64 for Basic Auth
        credentials = f"{self.app_key}:{self.app_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            "Accept": "application/json",
            "Authorization": f"Basic {encoded_credentials}",
            "posemesh-client-id": self._device_id
        }
        
        logger.debug(f"Requesting DDS token from: {url}")
        
        try:
            response = self.client.post(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            dds_token = data.get("access_token")
            
            if not dds_token:
                raise ValueError("No access_token in DDS token response")
            
            logger.debug("DDS token obtained successfully")
            return dds_token
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to get DDS token: {e}")
            raise ValueError(f"Failed to get app Domain Service access token: {e}")

    def auth(self):
        """Authenticate and get DDS token."""
        import time
        current_time = time.time()
        
        # Check if we have a valid cached DDS token
        if self._dds_token and self._dds_token_timestamp > 0:
            time_since_dds = current_time - self._dds_token_timestamp
            if time_since_dds < TOKEN_EXPIRATION_TIME:
                logger.debug(f"Using cached DDS token (age: {time_since_dds:.0f}s)")
                return True, ''
        
        # Get new DDS token
        try:
            self._dds_token = self._get_dds_token()
            self._dds_token_timestamp = current_time
            return True, ''
        except Exception as e:
            return False, str(e)

    def get_domain_id(self, qrshortcode):
        """Get Domain ID from QR shortcode."""
        if self.override_domain_id:
            return self.override_domain_id
        
        if not self._dds_token:
            ret, msg = self.auth()
            if not ret:
                return None
        
        url = f"{self.dds_base_url}/api/v1/lighthouses/{qrshortcode}/domains?org=all"
        
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._dds_token}",
            "posemesh-client-id": self._device_id
        }
        
        logger.debug(f"Requesting domain ID from: {url}")
        
        try:
            response = self.client.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            domain_id = self._extract_domain_id_from_response(data)
            return domain_id
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to get domain ID from lighthouses: {e}")
            return None

    def _extract_domain_id_from_response(self, domains_data: dict) -> Optional[str]:
        """Extract domain ID from domains response."""
        domains = domains_data.get("domains", [])
        
        if not domains:
            logger.warning("No domains found in response")
            return None
        
        # First, check for default domain
        for domain in domains:
            if domain.get("is_default", False):
                domain_id = domain.get("id")
                logger.debug(f"Found default domain ID: {domain_id}")
                return domain_id
        
        # If no default domain found, find the one with oldest added_to_domain_at
        oldest_domain = None
        oldest_date = None
        
        for domain in domains:
            added_at_str = domain.get("added_to_domain_at")
            if added_at_str:
                try:
                    # Parse the ISO format date
                    added_at_date = datetime.fromisoformat(added_at_str.replace('Z', '+00:00'))
                    
                    if oldest_date is None or added_at_date < oldest_date:
                        oldest_date = added_at_date
                        oldest_domain = domain
                except ValueError:
                    # Skip domains with invalid date format
                    continue
        
        if oldest_domain:
            domain_id = oldest_domain.get("id")
            logger.debug(f"Found oldest domain ID: {domain_id} (added at: {oldest_date})")
            return domain_id
        
        logger.warning("No valid domain found")
        return None

    def auth_domain(self, domain_id):
        """Authenticate Domain access."""
        if not self._dds_token:
            ret, msg = self.auth()
            if not ret:
                return False, 'Failed to authenticate - no DDS token'
        
        url = f"{self.dds_base_url}/api/v1/domains/{domain_id}/auth"
        
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._dds_token}",
            "posemesh-client-id": self._device_id
        }
        
        logger.debug(f"Authenticating domain at: {url}")
        
        try:
            response = self.client.post(url, headers=headers)
            response.raise_for_status()
            
            self._domain_info = response.json()
            import time
            self._domain_info_timestamp = time.time()
            
            domain_server_obj = self._domain_info.get("domain_server")
            if isinstance(domain_server_obj, dict):
                self._domain_server = domain_server_obj.get("url", "")
            elif isinstance(domain_server_obj, str):
                self._domain_server = domain_server_obj
            else:
                self._domain_server = None
            
            logger.debug("Domain authentication successful")
            return True, ''
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to authenticate domain access: {e}")
            return False, f'Failed to authenticate domain access: {e}'

    def fetch_portal_poses(self):
        """Fetch portal poses from the domain."""
        if not self._domain_info:
            return False, 'Domain Info not available'

        url = f"{self._domain_info['domain_server']['url']}/api/v1/domains/{self._domain_info['id']}/lighthouses"
        
        headers = {
            'authorization': f'Bearer {self._domain_info["access_token"]}',
            'posemesh-client-id': self._device_id
        }
        
        try:
            response = self.client.get(url, headers=headers)
            response.raise_for_status()
            
            response_json = response.json()
            
            for qr in response_json['poses']:
                trans = np.array([qr['px'], qr['py'], qr['pz']])
                quat = np.array([qr['rx'], qr['ry'], qr['rz'], qr['rw']])

                T_domain_portal = transformation_matrix(trans, quat)
                # Portal pose stays in domain frame - no conversion applied
                # if self._portal_conversion_matrix is not None:
                #     T_domain_portal = self._portal_conversion_matrix @ T_domain_portal

                portal = {
                    "short_id": qr["short_id"],
                    "size": float(qr['reported_size']) / 100.0,
                    "pose": T_domain_portal
                }
                self._portals[qr["short_id"]] = portal
            
            return True, ''
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch portal poses: {e}")
            return False, f'Failed to fetch the portals information: {e}'

    def portals(self):
        """Get the portals dictionary."""
        return self._portals

    def get_map(self, resolution=20, domain_id=None, image_format='png'):
        """
        Get map for the domain.
        
        :param resolution: Pixels per meter (default: 20)
        :param domain_id: Domain ID (default: None, uses current domain)
        :param image_format: Image format - 'png', 'bmp', 'pgm', or 'stcm' (default: 'png')
        :return: (image, yaml_dict) tuple, or (None, None) on error
        Note: Server only accepts 'png', so for 'pgm' we request PNG and convert
        """
        if domain_id is None:
            if not self._domain_info:
                return None, None
            domain_id = self._domain_info.get('id')
        
        if domain_id is None:
            return None, None
        
        if self._domain_server is None:
            logger.error("Domain server URL not set. Domain must be authenticated first.")
            return None, None

        url = self.map_endpoint
        headers = {
            'authorization': f'Bearer {self._domain_info["access_token"]}',
            'posemesh-client-id': self._device_id
        }

        # Server only accepts 'png', so request PNG even if we want PGM
        image_format_request = 'png' if image_format == 'pgm' else image_format
        
        body = {
            'domainId': domain_id,
            'domainServerUrl': self._domain_server,
            'height': 0.1,
            'pixelsPerMeter': resolution,
            'fileType': image_format_request,
        }

        try:
            logger.debug(f"Requesting map from: {url}")
            logger.debug(f"Request body: {body}")
            response = self.client.post(url, headers=headers, json=body)
            response.raise_for_status()
            raw_data = response.text

            # Split the data using the boundary marker
            boundary = raw_data.split("\n", 1)[0].strip()
            parts = raw_data.split(boundary)

            # Initialize placeholders for the image and YAML data
            image_data = None
            yaml_data = None

            # Iterate through each part of the form-data
            for part in parts:
                if "name=\"img\"" in part:
                    # Extract and decode the base64 image data, handle newlines
                    image_data_match = re.search(r"name=\"img\"\s*\n([a-zA-Z0-9+/=\n]+)", part)
                    if image_data_match:
                        # Remove any newlines or spaces in the base64-encoded data
                        encoded_image = "".join(image_data_match.group(1).splitlines())
                        image_data = base64.b64decode(encoded_image)
                elif "name=\"yaml\"" in part:
                    # Extract the YAML content
                    yaml_data_match = re.search(r"name=\"yaml\"\s*\n(.+)", part, re.DOTALL)
                    if yaml_data_match:
                        yaml_data = yaml_data_match.group(1).strip()

            image = Image.open(io.BytesIO(image_data))
            yaml_dict = yaml.safe_load(yaml_data)
            return image, yaml_dict
            
        except httpx.HTTPError as e:
            error_msg = f"Failed to get map: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_body = e.response.text
                    error_msg += f"\nResponse body: {error_body[:500]}"
                except:
                    pass
            logger.error(error_msg)
            return None, None

    def get_path(self, origin, target, domain_id=None):
        if domain_id is None:
            if not self._domain_info:
                return None, None
            domain_id = self._domain_info.get('id')
        
        if domain_id is None:
            return None, None
        
        if self._domain_server is None:
            logger.error("Domain server URL not set. Domain must be authenticated first.")
            return None, None

        method = 'POST'

        url = self.path_endpoint
        headers = {
            'authorization': f'Bearer {self._domain_info["access_token"]}',
            'posemesh-client-id': self._device_id
        }

        body = {
            'domainId': domain_id,
            'domainServerUrl': self._domain_server,
            'wayPoints': [origin, target],
            'radius': self.robot_radius,
            'optimizeRoute': False
        }
        success, response = send_request(method, url, headers, body)
        if not success:
            return None, None
        
        response_json = json.loads(response.text)

        return response_json['full']

    def get_restricted_dest(self, dest, domain_id=None):
        if domain_id is None:
            if not self._domain_info:
                return None, None
            domain_id = self._domain_info.get('id')
        
        if domain_id is None:
            return None, None
        
        if self._domain_server is None:
            logger.error("Domain server URL not set. Domain must be authenticated first.")
            return None, None

        method = 'POST'
        url = self.restricted_dest_endpoint
        headers = {'authorization': f'Bearer {self._domain_info["access_token"]}'}

        body = {
            'domainId': domain_id,
            'domainServerUrl': self._domain_server,
            'target': dest,
            'radius': 0.5
        }

        success, response = send_request(method, url, headers, body)

        if not success:
            return False, "Failed to get navmesh coord"

        x1 = dest['x']
        z1 = dest['z']
        x2 = response.json()['restricted']['x']
        z2 = response.json()['restricted']['z']

        delta_x = x1 - x2
        delta_z = z1 - z2

        z2 = -abs(z2) if z2 > 0 else abs(z2)
        pitch = round(math.atan2(delta_z, delta_x), 2)  # Result in radians, rounded to 2 decimal places
        pitch = -abs(pitch) if pitch > 0 else abs(pitch)
        pose = response.json()['restricted']
        pose['pitch'] = pitch

        return pose

    def close(self):
        """Close the HTTP client."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.client.close()
