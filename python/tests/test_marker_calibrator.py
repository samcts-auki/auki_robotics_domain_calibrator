"""Tests for marker_calibrator module."""

import unittest
import numpy as np
import cv2
from domain_calibrator import Calibrator, average_transforms


class TestMarkerCalibrator(unittest.TestCase):
    """Test cases for Calibrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calibrator = Calibrator()
    
    def test_preprocess_image_grayscale(self):
        """Test image preprocessing converts to grayscale."""
        # Create a test BGR image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        processed, scale = self.calibrator.preprocess_image(img, target_size=200)
        
        # Should be grayscale (2D)
        self.assertEqual(len(processed.shape), 2)
        # Should be resized
        self.assertLessEqual(max(processed.shape), 200)
    
    def test_preprocess_image_already_grayscale(self):
        """Test preprocessing already grayscale image."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        processed, scale = self.calibrator.preprocess_image(img, target_size=200)
        
        # Should still be grayscale
        self.assertEqual(len(processed.shape), 2)
    
    def test_camera_pose_success(self):
        """Test camera pose estimation with valid inputs."""
        # Create a mock portal
        portal = {
            'size': 0.1,  # 10cm
            'pose': np.eye(4)
        }
        
        # Create mock corners (4 corners of a square)
        corners = np.array([
            [100, 100],
            [200, 100],
            [200, 200],
            [100, 200]
        ], dtype=np.float32)
        
        # Create mock camera matrix
        camera_matrix = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros(4, dtype=np.float64)
        
        result = self.calibrator.camera_pose(portal, corners, camera_matrix, dist_coeffs)
        
        # Should return a 4x4 transformation matrix
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (4, 4))
    
    def test_camera_pose_failure(self):
        """Test camera pose estimation with invalid inputs."""
        portal = {'size': 0.1, 'pose': np.eye(4)}
        
        # Invalid corners (all same point)
        corners = np.array([
            [100, 100],
            [100, 100],
            [100, 100],
            [100, 100]
        ], dtype=np.float32)
        
        camera_matrix = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros(4, dtype=np.float64)
        
        result = self.calibrator.camera_pose(portal, corners, camera_matrix, dist_coeffs)
        
        # Should return None for invalid inputs
        self.assertIsNone(result)
    
    def test_average_transforms(self):
        """Test averaging multiple transforms."""
        # Create some test transforms
        transforms = [
            np.eye(4),
            np.eye(4),
            np.eye(4)
        ]
        
        result = average_transforms(transforms)
        
        # Should return a 4x4 matrix
        self.assertEqual(result.shape, (4, 4))
        # Should be close to identity for identity inputs
        np.testing.assert_array_almost_equal(result, np.eye(4), decimal=5)
    
    def test_average_transforms_with_translation(self):
        """Test averaging transforms with different translations."""
        transforms = [
            np.array([
                [1, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
            np.array([
                [1, 0, 0, 2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
            np.array([
                [1, 0, 0, 3],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        ]
        
        result = average_transforms(transforms)
        
        # Average translation should be 2.0
        self.assertAlmostEqual(result[0, 3], 2.0, places=5)


if __name__ == '__main__':
    unittest.main()
