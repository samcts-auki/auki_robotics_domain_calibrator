# Camera Calibration Utility

This utility helps you calibrate your camera to obtain the camera intrinsic matrix and distortion coefficients needed for accurate pose estimation.

## Prerequisites

You'll need a checkerboard pattern. You can:
- Print one from: https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
- Or create your own with a known square size

## Usage

### Option 1: Capture images from camera (interactive)

```bash
python utils/calibrate_camera.py \
    --checkerboard-rows 9 \
    --checkerboard-cols 6 \
    --square-size 0.025 \
    --camera-index 0 \
    --min-images 10
```

**Controls:**
- Press **SPACE** to capture an image when checkerboard is detected
- Press **'q'** to finish calibration

**Tips:**
- Move the checkerboard to different positions and angles
- Make sure the checkerboard is fully visible
- Capture images from various distances and orientations
- At least 10 images are recommended for good calibration

### Option 2: Use existing images

If you already have calibration images:

```bash
python utils/calibrate_camera.py \
    --images-dir path/to/calibration/images \
    --checkerboard-rows 9 \
    --checkerboard-cols 6 \
    --square-size 0.025
```

## Parameters

- `--checkerboard-rows`: Number of **inner corners** in rows (default: 9)
- `--checkerboard-cols`: Number of **inner corners** in columns (default: 6)
- `--square-size`: Size of each square in **meters** (default: 0.025 = 25mm)
- `--camera-index`: Camera device index (default: 0)
- `--images-dir`: Directory containing calibration images (optional)
- `--output-config`: Path to config.yaml file (default: config.yaml)
- `--min-images`: Minimum number of images required (default: 10)

## Example

For a 9x6 checkerboard with 25mm squares:

```bash
python utils/calibrate_camera.py \
    --checkerboard-rows 9 \
    --checkerboard-cols 6 \
    --square-size 0.025 \
    --min-images 15
```

## Output

The script will:
1. Calculate the camera intrinsic matrix
2. Calculate distortion coefficients
3. Display the reprojection error (lower is better, typically < 0.5 pixels)
4. Update `config.yaml` with the calibration results

The calibration results will be saved in `config.yaml`:

```yaml
camera:
  camera_matrix:
    - [fx, 0, cx]
    - [0, fy, cy]
    - [0, 0, 1]
  dist_coeffs: [k1, k2, p1, p2, k3]
```

## Troubleshooting

**"No checkerboard detected"**
- Ensure good lighting
- Make sure the checkerboard is flat and fully visible
- Check that the checkerboard size parameters match your pattern
- Try different angles and distances

**High reprojection error (> 1.0 pixels)**
- Capture more images (20-30 recommended)
- Ensure images are from diverse angles and positions
- Check that square size is accurate
- Verify checkerboard is flat and not warped

**Camera not opening**
- Check camera index (try 0, 1, 2, etc.)
- Ensure camera is not being used by another application
- On Linux, check permissions: `sudo chmod 666 /dev/video0`
