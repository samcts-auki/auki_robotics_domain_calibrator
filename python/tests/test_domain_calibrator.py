"""Tests for domain_calibrator module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from domain_calibrator import DomainCalibratorPy


# Note: rpy_from_matrix_scipy was removed, tests removed


class TestDomainCalibratorPy(unittest.TestCase):
    """Test cases for DomainCalibratorPy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.domain_config = {
            'app_key': 'test_key',
            'app_secret': 'test_secret',
            'api_base_url': 'https://api.test.com',
            'dds_base_url': 'https://dds.test.com',
            'map_endpoint': 'https://map.test.com'
        }
        
        self.camera_matrix = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros(4, dtype=np.float64)
    
    @patch('domain_calibrator.Domain')
    def test_init_success(self, mock_domain_class):
        """Test successful initialization."""
        mock_domain = MagicMock()
        mock_domain.auth.return_value = (True, '')
        mock_domain_class.return_value = mock_domain
        
        calibrator = DomainCalibratorPy(
            self.domain_config,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs
        )
        
        self.assertIsNotNone(calibrator.marker_calibrator)
        self.assertIsNotNone(calibrator.domain)
    
    @patch('domain_calibrator.Domain')
    def test_init_missing_camera_matrix(self, mock_domain_class):
        """Test initialization fails without camera matrix."""
        mock_domain = MagicMock()
        mock_domain.auth.return_value = (True, '')
        mock_domain_class.return_value = mock_domain
        
        calibrator = DomainCalibratorPy(
            self.domain_config,
            camera_matrix=None,
            dist_coeffs=self.dist_coeffs
        )
        
        # Should raise error when trying to use
        with self.assertRaises(ValueError):
            calibrator.detect_and_calibrate(np.zeros((100, 100, 3), dtype=np.uint8))
    
    @patch('domain_calibrator.Domain')
    @patch('domain_calibrator.Calibrator')
    def test_detect_and_calibrate_no_markers(self, mock_calibrator_class, mock_domain_class):
        """Test detect_and_calibrate with no markers detected."""
        mock_domain = MagicMock()
        mock_domain.auth.return_value = (True, '')
        mock_domain_class.return_value = mock_domain
        
        mock_calibrator = MagicMock()
        mock_calibrator.detect_markers.return_value = []
        mock_calibrator.preprocess_image.return_value = (np.zeros((100, 100), dtype=np.uint8), 1.0)
        mock_calibrator_class.return_value = mock_calibrator
        
        calibrator = DomainCalibratorPy(
            self.domain_config,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs
        )
        calibrator.marker_calibrator = mock_calibrator
        
        result = calibrator.detect_and_calibrate(np.zeros((100, 100, 3), dtype=np.uint8))
        
        self.assertIsNone(result)
    
    @patch('domain_calibrator.Domain')
    @patch('domain_calibrator.Calibrator')
    def test_detect_and_calibrate_invalid_qr(self, mock_calibrator_class, mock_domain_class):
        """Test detect_and_calibrate with invalid QR code."""
        mock_domain = MagicMock()
        mock_domain.auth.return_value = (True, '')
        mock_domain.get_domain_id.return_value = None
        mock_domain_class.return_value = mock_domain
        
        mock_calibrator = MagicMock()
        mock_calibrator.detect_markers.return_value = [
            {
                'value': 'INVALID_QR',
                'landmarks': [100, 100, 200, 100, 200, 200, 100, 200]
            }
        ]
        mock_calibrator.preprocess_image.return_value = (np.zeros((100, 100), dtype=np.uint8), 1.0)
        mock_calibrator_class.return_value = mock_calibrator
        
        calibrator = DomainCalibratorPy(
            self.domain_config,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs
        )
        calibrator.marker_calibrator = mock_calibrator
        
        result = calibrator.detect_and_calibrate(np.zeros((100, 100, 3), dtype=np.uint8))
        
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
