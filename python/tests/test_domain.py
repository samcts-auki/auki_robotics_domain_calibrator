"""Tests for domain module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from domain_calibrator import Domain
from domain_calibrator.domain import transformation_matrix


class TestTransformationMatrix(unittest.TestCase):
    """Test transformation_matrix function."""
    
    def test_transformation_matrix(self):
        """Test creating transformation matrix from translation and quaternion."""
        trans = np.array([1.0, 2.0, 3.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        
        T = transformation_matrix(trans, quat)
        
        # Should be 4x4
        self.assertEqual(T.shape, (4, 4))
        # Translation should match
        np.testing.assert_array_almost_equal(T[:3, 3], trans)
        # Rotation should be identity
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))


class TestDomain(unittest.TestCase):
    """Test cases for Domain class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.domain_config = {
            'app_key': 'test_key',
            'app_secret': 'test_secret',
            'api_base_url': 'https://api.test.com',
            'dds_base_url': 'https://dds.test.com',
            'map_endpoint': 'https://map.test.com'
        }
    
    def test_init_missing_credentials(self):
        """Test initialization fails without app_key or app_secret."""
        with self.assertRaises(ValueError):
            Domain({})
        
        with self.assertRaises(ValueError):
            Domain({'app_key': 'key'})
        
        with self.assertRaises(ValueError):
            Domain({'app_secret': 'secret'})
    
    @patch('domain_calibrator.domain.httpx.Client')
    def test_init_success(self, mock_client):
        """Test successful initialization."""
        domain = Domain(self.domain_config)
        
        self.assertEqual(domain.app_key, 'test_key')
        self.assertEqual(domain.app_secret, 'test_secret')
        self.assertIsNotNone(domain.client)
    
    @patch('domain_calibrator.domain.httpx.Client')
    def test_get_dds_token(self, mock_client_class):
        """Test getting DDS token."""
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {'access_token': 'test_token'}
        mock_response.raise_for_status = Mock()
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        domain = Domain(self.domain_config)
        # Clear any calls from __init__ (auth() might call it)
        mock_client.post.reset_mock()
        token = domain._get_dds_token()
        
        self.assertEqual(token, 'test_token')
        # Should be called at least once (might be called during auth in __init__)
        self.assertGreaterEqual(mock_client.post.call_count, 1)
    
    @patch('domain_calibrator.domain.httpx.Client')
    def test_get_domain_id(self, mock_client_class):
        """Test getting domain ID from QR shortcode."""
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'domains': [
                {'id': 'domain1', 'is_default': True},
                {'id': 'domain2', 'is_default': False}
            ]
        }
        mock_response.raise_for_status = Mock()
        
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.post.return_value = MagicMock(
            json=lambda: {'access_token': 'test_token'},
            raise_for_status=Mock()
        )
        mock_client_class.return_value = mock_client
        
        domain = Domain(self.domain_config)
        domain_id = domain.get_domain_id('test_qr')
        
        self.assertEqual(domain_id, 'domain1')
    
    @patch('domain_calibrator.domain.httpx.Client')
    def test_fetch_portal_poses(self, mock_client_class):
        """Test fetching portal poses."""
        # Mock domain info
        mock_domain_response = MagicMock()
        mock_domain_response.json.return_value = {
            'id': 'domain1',
            'access_token': 'token',
            'domain_server': {'url': 'https://server.test.com'}
        }
        mock_domain_response.raise_for_status = Mock()
        
        # Mock portal response
        mock_portal_response = MagicMock()
        mock_portal_response.json.return_value = {
            'poses': [
                {
                    'short_id': 'QR123',
                    'px': 1.0, 'py': 2.0, 'pz': 3.0,
                    'rx': 0.0, 'ry': 0.0, 'rz': 0.0, 'rw': 1.0,
                    'reported_size': 100
                }
            ]
        }
        mock_portal_response.raise_for_status = Mock()
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_domain_response
        mock_client.get.side_effect = [mock_portal_response]
        mock_client_class.return_value = mock_client
        
        domain = Domain(self.domain_config)
        domain._domain_info = {
            'id': 'domain1',
            'access_token': 'token',
            'domain_server': {'url': 'https://server.test.com'}
        }
        
        success, msg = domain.fetch_portal_poses()
        
        self.assertTrue(success)
        self.assertIn('QR123', domain.portals())
        portal = domain.portals()['QR123']
        self.assertEqual(portal['short_id'], 'QR123')
        self.assertEqual(portal['size'], 1.0)  # 100 / 100


if __name__ == '__main__':
    unittest.main()
