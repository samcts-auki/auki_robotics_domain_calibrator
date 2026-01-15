# Domain Calibrator (Python)

Python implementation of domain calibrator using zbar for QR code detection.

## Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the package in development mode:
```bash
pip install -e .
```

This will install the `domain_calibrator` package and all its dependencies.

For development dependencies (pytest, etc.):
```bash
pip install -e ".[dev]"
```

## Configuration

1. Copy the example configuration file:
```bash
cp config.yaml.example config.yaml
```

2. Edit `config.yaml` to set your authentication credentials:
- `app_key`: Your application key
- `app_secret`: Your application secret
- `camera_matrix`: 3x3 camera intrinsic matrix (will be set by camera calibration)
- `dist_coeffs`: Camera distortion coefficients (will be set by camera calibration)

**Note:** `config.yaml` contains sensitive credentials and is excluded from git. Always use `config.yaml.example` as a template.

## Camera Calibration

Before using the domain calibrator, you need to calibrate your camera to obtain the camera matrix and distortion coefficients. Use the calibration utility:

```bash
python utils/calibrate_camera.py \
    --checkerboard-rows 9 \
    --checkerboard-cols 6 \
    --square-size 0.025 \
    --min-images 10
```

This will automatically update `config.yaml` with the calibration results. See `utils/README.md` for detailed instructions.

## Usage

### Example Script

Run the example script to see the domain calibrator in action:

```bash
# Use default camera from config.yaml
python example.py

# Override camera index
python example.py --camera-index 1

# Use camera path (Linux)
python example.py --camera-path /dev/video0
```

This will:
- Open your camera (default from `config.yaml`, or override via command-line)
- Detect QR codes at 2 Hz
- On first detection (or when domain changes):
  - Authenticate with the domain
  - Fetch the map (PGM + YAML)
  - List all portals in the domain
  - Report the camera pose

**Camera Configuration:**
- Set default camera in `config.yaml` under `camera.device` (integer index or string path)
- Override via `--camera-index` (integer) or `--camera-path` (string) command-line arguments
- `--camera-path` takes precedence over `--camera-index` if both are provided

### Programmatic Usage

```python
from domain_calibrator import DomainCalibratorPy
import cv2
import numpy as np

# Option 1: Load from config.yaml file (default)
calibrator = DomainCalibratorPy()

# Option 2: Provide config programmatically
domain_config = {
    'app_key': 'your_key',
    'app_secret': 'your_secret',
    'api_base_url': 'https://api.auki.network',
    'dds_base_url': 'https://dds.auki.network',
    'map_endpoint': 'https://dsc.dev.aukiverse.com/spatial/crosssection'
}

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float64)

calibrator = DomainCalibratorPy(
    domain_config=domain_config,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

# Detect and calibrate from image
image = cv2.imread('image.jpg')
transform = calibrator.detect_and_calibrate(image)

if transform is not None:
    print(f"Camera pose: {transform}")
```

## Running Tests

```bash
pytest tests/
```

## Dependencies

- numpy
- opencv-python
- scipy
- Pillow
- pyyaml
- requests
- httpx
- pyzbar (requires zbar library to be installed on system)

### Installing zbar

**macOS:**
```bash
brew install zbar
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libzbar0
```

**Windows:**
Download from: https://github.com/mchehab/zbar/releases
