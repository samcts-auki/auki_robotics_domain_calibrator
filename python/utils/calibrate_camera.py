#!/usr/bin/env python3
"""
Camera Calibration Utility

This script performs camera calibration using a checkerboard pattern.
It calculates the camera intrinsic matrix and distortion coefficients,
then updates the config.yaml file with the results.

Usage:
    python utils/calibrate_camera.py [options]

Options:
    --checkerboard-rows ROWS    Number of inner corners in checkerboard rows (default: 9)
    --checkerboard-cols COLS    Number of inner corners in checkerboard cols (default: 6)
    --square-size SIZE          Size of each square in meters (default: 0.025)
    --camera-index INDEX        Camera device index (default: 0)
    --images-dir DIR            Directory containing calibration images (optional)
    --output-config PATH        Path to config.yaml file (default: config.yaml)
    --min-images MIN            Minimum number of images required (default: 10)
"""

import cv2
import numpy as np
import yaml
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional


class CameraCalibrator:
    """Camera calibration using checkerboard pattern."""
    
    def __init__(self, checkerboard_size: Tuple[int, int], square_size: float):
        """
        Initialize the calibrator.
        
        Args:
            checkerboard_size: (rows, cols) - number of inner corners
            square_size: Size of each square in meters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Arrays to store object points and image points from all images
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane
        
    def find_checkerboard(self, img: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Find checkerboard corners in image.
        
        Args:
            img: Input image (grayscale)
            
        Returns:
            (corners, objp) if found, None otherwise
        """
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            img, 
            self.checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            return corners2, self.objp
        return None
    
    def add_image(self, img: np.ndarray) -> bool:
        """
        Add an image for calibration.
        
        Args:
            img: Input image (can be color or grayscale)
            
        Returns:
            True if checkerboard was found, False otherwise
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        result = self.find_checkerboard(gray)
        if result:
            corners, objp = result
            self.objpoints.append(objp)
            self.imgpoints.append(corners)
            return True
        return False
    
    def calibrate(self, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform camera calibration.
        
        Args:
            image_size: (width, height) of images
            
        Returns:
            (camera_matrix, dist_coeffs, reprojection_error)
        """
        if len(self.objpoints) < 3:
            raise ValueError(f"Need at least 3 images with detected checkerboards. Found {len(self.objpoints)}")
        
        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            image_size,
            None,
            None
        )
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(self.objpoints)
        
        return camera_matrix, dist_coeffs, mean_error


def capture_images_from_camera(calibrator: CameraCalibrator, camera_index: int = 0, min_images: int = 10) -> Tuple[int, Tuple[int, int]]:
    """
    Capture images from camera and detect checkerboards.
    
    Args:
        calibrator: CameraCalibrator instance
        camera_index: Camera device index
        min_images: Minimum number of images to capture
        
    Returns:
        (number_of_valid_images, image_size)
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    
    print(f"\nCamera calibration - Press SPACE to capture, 'q' to finish")
    print(f"Need at least {min_images} images with detected checkerboards")
    print(f"Current: 0/{min_images}")
    
    image_size = None
    count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Try to find checkerboard
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            result = calibrator.find_checkerboard(gray)
            
            # Draw checkerboard if found
            display_frame = frame.copy()
            if result:
                corners, _ = result
                cv2.drawChessboardCorners(display_frame, calibrator.checkerboard_size, corners, True)
                status = f"Checkerboard found! ({count}/{min_images}) - Press SPACE to capture"
                color = (0, 255, 0)
            else:
                status = f"No checkerboard detected ({count}/{min_images}) - Move checkerboard"
                color = (0, 0, 255)
            
            cv2.putText(display_frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display_frame, "Press 'q' to finish", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Camera Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar
                if result:
                    if calibrator.add_image(frame):
                        count += 1
                        image_size = (frame.shape[1], frame.shape[0])
                        print(f"Captured image {count}/{min_images}")
                        if count >= min_images:
                            print(f"Minimum images reached! You can continue or press 'q' to finish.")
                else:
                    print("Checkerboard not detected in this frame. Try again.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    if image_size is None:
        raise RuntimeError("No images captured")
    
    return count, image_size


def load_images_from_directory(images_dir: str, calibrator: CameraCalibrator) -> Tuple[int, Tuple[int, int]]:
    """
    Load images from directory and detect checkerboards.
    
    Args:
        images_dir: Directory containing calibration images
        calibrator: CameraCalibrator instance
        
    Returns:
        (number_of_valid_images, image_size)
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    print(f"Loading {len(image_files)} images from {images_dir}")
    
    count = 0
    image_size = None
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Could not load {img_file}")
            continue
        
        if calibrator.add_image(img):
            count += 1
            image_size = (img.shape[1], img.shape[0])
            print(f"✓ {img_file.name} - checkerboard found ({count} total)")
        else:
            print(f"✗ {img_file.name} - no checkerboard detected")
    
    if image_size is None:
        raise RuntimeError("No valid images with checkerboards found")
    
    return count, image_size


def update_config_file(config_path: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    """
    Update config.yaml with camera calibration results.
    
    Args:
        config_path: Path to config.yaml file
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    """
    config_path = Path(config_path)
    
    # Load existing config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    
    # Ensure camera section exists
    if 'camera' not in config:
        config['camera'] = {}
    
    # Update camera matrix (convert to list for YAML)
    config['camera']['camera_matrix'] = camera_matrix.tolist()
    
    # Update distortion coefficients
    config['camera']['dist_coeffs'] = dist_coeffs.flatten().tolist()
    
    # Write back to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Updated {config_path} with camera calibration results")


def main():
    parser = argparse.ArgumentParser(
        description='Camera calibration utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--checkerboard-rows', type=int, default=9,
                       help='Number of inner corners in checkerboard rows (default: 9)')
    parser.add_argument('--checkerboard-cols', type=int, default=6,
                       help='Number of inner corners in checkerboard cols (default: 6)')
    parser.add_argument('--square-size', type=float, default=0.025,
                       help='Size of each square in meters (default: 0.025)')
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--images-dir', type=str, default=None,
                       help='Directory containing calibration images (optional)')
    parser.add_argument('--output-config', type=str, default='config.yaml',
                       help='Path to config.yaml file (default: config.yaml)')
    parser.add_argument('--min-images', type=int, default=10,
                       help='Minimum number of images required (default: 10)')
    
    args = parser.parse_args()
    
    # Create calibrator
    checkerboard_size = (args.checkerboard_rows, args.checkerboard_cols)
    calibrator = CameraCalibrator(checkerboard_size, args.square_size)
    
    print("=" * 60)
    print("Camera Calibration Utility")
    print("=" * 60)
    print(f"Checkerboard: {args.checkerboard_rows}x{args.checkerboard_cols} inner corners")
    print(f"Square size: {args.square_size} meters")
    print("=" * 60)
    
    # Capture or load images
    if args.images_dir:
        count, image_size = load_images_from_directory(args.images_dir, calibrator)
    else:
        count, image_size = capture_images_from_camera(calibrator, args.camera_index, args.min_images)
    
    if count < 3:
        print(f"\n✗ Error: Need at least 3 images with checkerboards. Found {count}")
        sys.exit(1)
    
    print(f"\n✓ Found {count} images with checkerboards")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    
    # Perform calibration
    print("\nPerforming calibration...")
    try:
        camera_matrix, dist_coeffs, reprojection_error = calibrator.calibrate(image_size)
        
        print("\n" + "=" * 60)
        print("Calibration Results")
        print("=" * 60)
        print(f"Camera Matrix:")
        print(camera_matrix)
        print(f"\nDistortion Coefficients:")
        print(dist_coeffs.flatten())
        print(f"\nReprojection Error: {reprojection_error:.4f} pixels")
        print("=" * 60)
        
        # Update config file
        config_path = Path(args.output_config)
        if not config_path.is_absolute():
            # Make relative to script location or current directory
            script_dir = Path(__file__).parent.parent
            config_path = script_dir / config_path
        
        update_config_file(str(config_path), camera_matrix, dist_coeffs)
        
        print("\n✓ Calibration complete!")
        
    except Exception as e:
        print(f"\n✗ Calibration failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
