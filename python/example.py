#!/usr/bin/env python3
"""
Example usage of the domain_calibrator package.

This script demonstrates how to use DomainCalibratorPy to:
- Detect QR codes from a camera feed
- Fetch domain maps (PGM + YAML)
- Get portal poses
- Calculate camera pose in domain coordinates

Usage:
    python example.py [--camera-index INDEX] [--camera-path PATH]
    
    --camera-index: Camera device index (integer, default: from config.yaml)
    --camera-path:  Camera device path (string, e.g., "/dev/video0")
    
    If both are provided, --camera-path takes precedence.
"""

import cv2
import time
import yaml
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from domain_calibrator import DomainCalibratorPy, load_config


def main():
    """
    Main function that runs continuously, checking camera at 2 Hz.
    On first QR detection (or when domain changes):
    - Get the domain_id
    - Fetch the map (pgm and yaml files)
    - Get the list of Portals (with their pose)
    - Report the camera pose based on the QR that was seen.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Domain Calibrator Example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--camera-index', type=int, default=None,
                       help='Camera device index (integer, overrides config)')
    parser.add_argument('--camera-path', type=str, default=None,
                       help='Camera device path (string, e.g., "/dev/video0", overrides config and --camera-index)')
    args = parser.parse_args()
    
    # Load config to get default camera device
    config = load_config()
    default_camera_device = config.get('camera', {}).get('device', 0)
    
    # Determine camera device (command-line takes precedence)
    if args.camera_path:
        camera_device = args.camera_path
        print(f"Using camera path: {camera_device}")
    elif args.camera_index is not None:
        camera_device = args.camera_index
        print(f"Using camera index: {camera_device}")
    else:
        camera_device = default_camera_device
        if isinstance(camera_device, str):
            print(f"Using camera path from config: {camera_device}")
        else:
            print(f"Using camera index from config: {camera_device}")
    
    # Initialize calibrator (loads config from config.yaml)
    calibrator = DomainCalibratorPy()
    
    # Open camera
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened():
        print(f"Failed to open camera: {camera_device}")
        raise SystemExit(1)
    
    print("Domain Calibrator running. Checking camera at 2 Hz.")
    print("Waiting for QR code detection...")
    
    initialized_domain_id = None  # Track which domain has been initialized
    check_interval = 0.5  # 2 Hz = 0.5 seconds between checks
    last_check_time = 0.0
    
    # Get the directory where this script is located (for saving map files)
    script_dir = Path(__file__).parent
    
    try:
        while True:
            current_time = time.time()
            
            # Check camera at 2 Hz
            if current_time - last_check_time < check_interval:
                time.sleep(0.01)  # Small sleep to avoid busy waiting
                continue
            
            last_check_time = current_time
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                continue
            
            # Detect and calibrate
            result = calibrator.detect_and_calibrate(frame)
            
            # Check if we need to initialize (first detection or different domain)
            # After detect_and_calibrate, the domain should be authenticated if a QR was found
            current_domain_id = calibrator._authenticated_domain_id
            needs_initialization = False
            
            if result is not None:
                # Check if this is a new domain or first initialization
                if initialized_domain_id is None:
                    needs_initialization = True
                elif current_domain_id != initialized_domain_id:
                    needs_initialization = True
                    print(f"\n⚠️  Domain changed: {initialized_domain_id} -> {current_domain_id}")
            
            if result is not None and needs_initialization:
                # Get domain_id (from the last detected QR)
                domain_id = calibrator._authenticated_domain_id
                if domain_id is None:
                    print("Warning: No domain_id available")
                    continue
                
                # Check if this is a new domain or first detection
                if initialized_domain_id is None:
                    print(f"\n{'='*60}")
                    print(f"FIRST QR DETECTION - Initializing domain")
                    print(f"{'='*60}")
                else:
                    print(f"\n{'='*60}")
                    print(f"NEW DOMAIN DETECTED - Reinitializing")
                    print(f"{'='*60}")
                    print(f"Previous domain: {initialized_domain_id}")
                
                print(f"Domain ID: {domain_id}")
                initialized_domain_id = domain_id
                
                # Fetch map (server only supports PNG, we'll convert to PGM)
                print("\nFetching map...")
                map_image, map_yaml = calibrator.get_map(resolution=20, domain_id=domain_id, image_format='png')
                
                if map_image is not None and map_yaml is not None:
                    # Save map as PGM (convert from PNG)
                    map_pgm_path = script_dir / "map.pgm"
                    # Convert to grayscale
                    if map_image.mode != 'L':
                        map_image = map_image.convert('L')
                    
                    width, height = map_image.size
                    
                    # Create a binary occupancy grid: 0 (free/black), 255 (occupied/white), 128 (unknown/gray)
                    binary_grid = []
                    for pixel in map_image.getdata():
                        if pixel > 165:  # Occupied threshold (65% of 255)
                            binary_grid.append(255)  # Occupied (white)
                        elif pixel < 50:  # Free threshold (19.6% of 255)
                            binary_grid.append(0)  # Free (black)
                        else:
                            binary_grid.append(128)  # Unknown (gray)
                    
                    # Save as PGM P2 format (ASCII)
                    with open(map_pgm_path, 'w') as f:
                        f.write(f"P2\n{width} {height}\n255\n")
                        # Write in rows
                        for i in range(0, len(binary_grid), width):
                            row = binary_grid[i:i + width]
                            f.write(" ".join(map(str, row)) + "\n")
                    print(f"✓ Saved map image to: {map_pgm_path}")
                    
                    # Save map YAML (update image field to reference PGM)
                    map_yaml_path = script_dir / "map.yaml"
                    # Update YAML to reference the PGM file
                    map_yaml['image'] = 'map.pgm'
                    with open(map_yaml_path, 'w') as f:
                        yaml.dump(map_yaml, f, default_flow_style=False)
                    print(f"✓ Saved map YAML to: {map_yaml_path}")
                else:
                    print("✗ Failed to fetch map")
                
                # Get list of portals (already fetched during detect_and_calibrate)
                portals = calibrator.domain.portals()
                print(f"\nPortals in domain ({len(portals)} total):")
                for portal_id, portal in portals.items():
                    portal_pos = portal['pose'][:3, 3]
                    print(f"  {portal_id}: position ({portal_pos[0]:.3f}, {portal_pos[1]:.3f}, {portal_pos[2]:.3f}) m, size: {portal['size']:.3f} m")
                
                # Report camera pose
                camera_pos = result[:3, 3]
                camera_rot_matrix = result[:3, :3]
                camera_rot = R.from_matrix(camera_rot_matrix)
                camera_euler_zyx = camera_rot.as_euler('zyx', degrees=True)
                
                print(f"\n{'='*60}")
                print(f"CAMERA POSE")
                print(f"{'='*60}")
                print(f"Position (x, y, z): ({camera_pos[0]:.4f}, {camera_pos[1]:.4f}, {camera_pos[2]:.4f}) m")
                print(f"Rotation:")
                print(f"  Yaw:   {camera_euler_zyx[0]:.2f}°")
                print(f"  Pitch: {camera_euler_zyx[1]:.2f}°")
                print(f"  Roll:  {camera_euler_zyx[2]:.2f}°")
                print(f"{'='*60}\n")
                
            elif result is not None:
                # Subsequent detections - just report pose
                camera_pos = result[:3, 3]
                camera_rot_matrix = result[:3, :3]
                camera_rot = R.from_matrix(camera_rot_matrix)
                camera_euler_zyx = camera_rot.as_euler('zyx', degrees=True)
                
                print(f"Camera pose: pos=({camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}) m, "
                      f"yaw={camera_euler_zyx[0]:.1f}°, pitch={camera_euler_zyx[1]:.1f}°, roll={camera_euler_zyx[2]:.1f}°")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        print("Camera released")


if __name__ == "__main__":
    main()
